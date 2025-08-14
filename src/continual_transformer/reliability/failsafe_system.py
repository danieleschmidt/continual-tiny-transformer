"""
Failsafe and recovery system for continual learning models.
Provides automatic recovery, graceful degradation, and system resilience.
"""

import time
import threading
import logging
import torch
import pickle
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
from enum import Enum
import traceback
import copy

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of system failures."""
    MEMORY_OVERFLOW = "memory_overflow"
    CUDA_ERROR = "cuda_error"
    MODEL_ERROR = "model_error"
    INFERENCE_TIMEOUT = "inference_timeout"
    TRAINING_DIVERGENCE = "training_divergence"
    DATA_CORRUPTION = "data_corruption"
    HARDWARE_FAILURE = "hardware_failure"
    UNKNOWN = "unknown"


@dataclass
class FailureEvent:
    """Record of a system failure."""
    timestamp: float
    failure_type: FailureType
    component: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None


@dataclass
class RecoveryStrategy:
    """Definition of a recovery strategy."""
    name: str
    failure_types: List[FailureType]
    priority: int  # Lower number = higher priority
    max_attempts: int
    cooldown_seconds: float
    recovery_function: Callable


class FailsafeSystem:
    """Comprehensive failsafe and recovery system."""
    
    def __init__(self, model, config, checkpoint_interval: float = 300.0):
        self.model = model
        self.config = config
        self.checkpoint_interval = checkpoint_interval
        
        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = []
        self._register_default_strategies()
        
        # Failure tracking
        self.failure_history = deque(maxlen=1000)
        self.recovery_attempts = defaultdict(int)
        self.last_recovery_time = defaultdict(float)
        
        # System state
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_checkpoint_time = time.time()
        
        # Checkpoints and backups
        self.checkpoint_dir = Path(config.output_dir) / "checkpoints" if hasattr(config, 'output_dir') else Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Circuit breaker state
        self.circuit_breaker_state = defaultdict(lambda: {"failures": 0, "last_failure": 0, "state": "closed"})
        self.circuit_breaker_thresholds = {
            "failure_count": 5,
            "time_window": 300,  # 5 minutes
            "recovery_timeout": 60  # 1 minute
        }
        
        # Performance baselines
        self.performance_baselines = {}
        
        logger.info("FailsafeSystem initialized")
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        
        # Memory recovery strategy
        self.register_recovery_strategy(RecoveryStrategy(
            name="memory_cleanup",
            failure_types=[FailureType.MEMORY_OVERFLOW, FailureType.CUDA_ERROR],
            priority=1,
            max_attempts=3,
            cooldown_seconds=10.0,
            recovery_function=self._recover_memory_overflow
        ))
        
        # Model reset strategy
        self.register_recovery_strategy(RecoveryStrategy(
            name="model_reset",
            failure_types=[FailureType.MODEL_ERROR, FailureType.TRAINING_DIVERGENCE],
            priority=2,
            max_attempts=2,
            cooldown_seconds=30.0,
            recovery_function=self._recover_model_error
        ))
        
        # Device fallback strategy
        self.register_recovery_strategy(RecoveryStrategy(
            name="device_fallback",
            failure_types=[FailureType.CUDA_ERROR, FailureType.HARDWARE_FAILURE],
            priority=3,
            max_attempts=1,
            cooldown_seconds=0.0,
            recovery_function=self._recover_device_fallback
        ))
        
        # Checkpoint restore strategy
        self.register_recovery_strategy(RecoveryStrategy(
            name="checkpoint_restore",
            failure_types=[FailureType.MODEL_ERROR, FailureType.DATA_CORRUPTION],
            priority=4,
            max_attempts=1,
            cooldown_seconds=0.0,
            recovery_function=self._recover_from_checkpoint
        ))
        
        # Graceful degradation strategy
        self.register_recovery_strategy(RecoveryStrategy(
            name="graceful_degradation",
            failure_types=[FailureType.INFERENCE_TIMEOUT, FailureType.MEMORY_OVERFLOW],
            priority=5,
            max_attempts=1,
            cooldown_seconds=0.0,
            recovery_function=self._recover_graceful_degradation
        ))
    
    def register_recovery_strategy(self, strategy: RecoveryStrategy):
        """Register a new recovery strategy."""
        self.recovery_strategies.append(strategy)
        self.recovery_strategies.sort(key=lambda s: s.priority)
        logger.info(f"Registered recovery strategy: {strategy.name}")
    
    def start_monitoring(self):
        """Start background monitoring for failures."""
        if self.is_monitoring:
            logger.warning("Failsafe monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Create initial checkpoint
        self._create_checkpoint("initial")
        
        logger.info("Failsafe monitoring started")
    
    def stop_monitoring(self):
        """Stop failsafe monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Failsafe monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for automated checkpointing."""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Create periodic checkpoints
                if current_time - self.last_checkpoint_time > self.checkpoint_interval:
                    self._create_checkpoint("periodic")
                    self.last_checkpoint_time = current_time
                
                # Update circuit breaker states
                self._update_circuit_breakers()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in failsafe monitoring loop: {e}")
                time.sleep(30)
    
    def handle_failure(
        self,
        exception: Exception,
        context: Dict[str, Any],
        component: str = "unknown"
    ) -> Tuple[bool, str]:
        """
        Handle a system failure and attempt recovery.
        
        Returns:
            Tuple of (recovery_successful, recovery_message)
        """
        
        # Classify the failure
        failure_type = self._classify_failure(exception, context)
        
        # Record the failure
        failure_event = FailureEvent(
            timestamp=time.time(),
            failure_type=failure_type,
            component=component,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            context=context.copy()
        )
        
        self.failure_history.append(failure_event)
        
        logger.error(f"Failure detected: {failure_type.value} in {component} - {str(exception)}")
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(component):
            logger.warning(f"Circuit breaker open for {component}, skipping recovery")
            return False, "Circuit breaker open"
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(failure_event)
        
        # Update circuit breaker based on result
        if not recovery_result[0]:
            self._record_circuit_breaker_failure(component)
        
        return recovery_result
    
    def _classify_failure(self, exception: Exception, context: Dict[str, Any]) -> FailureType:
        """Classify the type of failure based on exception and context."""
        
        error_message = str(exception).lower()
        exception_type = type(exception).__name__
        
        # Memory-related errors
        if ("memory" in error_message or "oom" in error_message or 
            exception_type in ["RuntimeError"] and "memory" in error_message):
            return FailureType.MEMORY_OVERFLOW
        
        # CUDA errors
        if ("cuda" in error_message or "gpu" in error_message or
            exception_type in ["CudaError", "CudaOutOfMemoryError"]):
            return FailureType.CUDA_ERROR
        
        # Model-specific errors
        if ("model" in error_message or "forward" in error_message or
            exception_type in ["ValueError", "AttributeError"] and 
            any(k in context for k in ["task_id", "model_state"])):
            return FailureType.MODEL_ERROR
        
        # Training divergence
        if ("nan" in error_message or "inf" in error_message or
            context.get("loss") == float('inf') or context.get("loss") != context.get("loss")):
            return FailureType.TRAINING_DIVERGENCE
        
        # Timeout errors
        if "timeout" in error_message or exception_type == "TimeoutError":
            return FailureType.INFERENCE_TIMEOUT
        
        # Hardware failures
        if ("hardware" in error_message or "device" in error_message):
            return FailureType.HARDWARE_FAILURE
        
        return FailureType.UNKNOWN
    
    def _attempt_recovery(self, failure_event: FailureEvent) -> Tuple[bool, str]:
        """Attempt to recover from a failure using registered strategies."""
        
        failure_event.recovery_attempted = True
        
        # Find applicable recovery strategies
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies
            if failure_event.failure_type in strategy.failure_types
        ]
        
        if not applicable_strategies:
            logger.warning(f"No recovery strategies for failure type: {failure_event.failure_type}")
            return False, "No applicable recovery strategies"
        
        # Try each strategy
        for strategy in applicable_strategies:
            # Check if we've exceeded max attempts
            if self.recovery_attempts[strategy.name] >= strategy.max_attempts:
                continue
            
            # Check cooldown
            time_since_last = time.time() - self.last_recovery_time[strategy.name]
            if time_since_last < strategy.cooldown_seconds:
                continue
            
            logger.info(f"Attempting recovery with strategy: {strategy.name}")
            
            try:
                # Attempt recovery
                self.recovery_attempts[strategy.name] += 1
                self.last_recovery_time[strategy.name] = time.time()
                
                success = strategy.recovery_function(failure_event)
                
                if success:
                    failure_event.recovery_successful = True
                    failure_event.recovery_method = strategy.name
                    
                    logger.info(f"Recovery successful with strategy: {strategy.name}")
                    
                    # Reset attempt counter on success
                    self.recovery_attempts[strategy.name] = 0
                    
                    return True, f"Recovered using {strategy.name}"
                else:
                    logger.warning(f"Recovery failed with strategy: {strategy.name}")
                
            except Exception as e:
                logger.error(f"Recovery strategy {strategy.name} raised exception: {e}")
        
        logger.error(f"All recovery strategies failed for {failure_event.failure_type}")
        return False, "All recovery strategies failed"
    
    def _recover_memory_overflow(self, failure_event: FailureEvent) -> bool:
        """Recover from memory overflow."""
        
        try:
            logger.info("Attempting memory cleanup...")
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reduce batch size if in context
            if "batch_size" in failure_event.context:
                new_batch_size = max(1, failure_event.context["batch_size"] // 2)
                logger.info(f"Reducing batch size from {failure_event.context['batch_size']} to {new_batch_size}")
                
                # This would need to be implemented based on how the model stores batch size
                if hasattr(self.config, 'batch_size'):
                    self.config.batch_size = new_batch_size
            
            # Enable gradient checkpointing if available
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False
    
    def _recover_model_error(self, failure_event: FailureEvent) -> bool:
        """Recover from model errors."""
        
        try:
            logger.info("Attempting model error recovery...")
            
            # Reset model to eval mode
            self.model.eval()
            
            # Clear any cached states
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
            
            # Reinitialize problematic components
            task_id = failure_event.context.get("task_id")
            if task_id and hasattr(self.model, 'adapters') and task_id in self.model.adapters:
                logger.info(f"Reinitializing adapter for task: {task_id}")
                
                # This would need specific implementation based on adapter architecture
                # Placeholder for adapter reinitialization
                
            return True
            
        except Exception as e:
            logger.error(f"Model error recovery failed: {e}")
            return False
    
    def _recover_device_fallback(self, failure_event: FailureEvent) -> bool:
        """Recover by falling back to CPU."""
        
        try:
            if self.config.device.type == "cuda":
                logger.info("Falling back to CPU device...")
                
                # Move model to CPU
                self.model.cpu()
                self.config.device = torch.device("cpu")
                
                # Disable CUDA optimizations
                if hasattr(self.config, 'mixed_precision'):
                    self.config.mixed_precision = False
                
                logger.info("Successfully moved to CPU")
                return True
            else:
                logger.warning("Already on CPU, cannot fall back further")
                return False
                
        except Exception as e:
            logger.error(f"Device fallback failed: {e}")
            return False
    
    def _recover_from_checkpoint(self, failure_event: FailureEvent) -> bool:
        """Recover by restoring from the latest checkpoint."""
        
        try:
            logger.info("Attempting checkpoint recovery...")
            
            # Find latest checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoint_files:
                logger.warning("No checkpoints available for recovery")
                return False
            
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            logger.info(f"Restoring from checkpoint: {latest_checkpoint}")
            
            # Load checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location=self.config.device)
            
            # Restore model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Restore other state if available
            if "task_router_mappings" in checkpoint:
                mappings = checkpoint["task_router_mappings"]
                if hasattr(self.model, 'task_router'):
                    self.model.task_router.task_id_to_index = mappings["task_id_to_index"]
                    self.model.task_router.index_to_task_id = mappings["index_to_task_id"]
                    self.model.task_router.num_tasks = mappings["num_tasks"]
            
            logger.info("Checkpoint recovery successful")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint recovery failed: {e}")
            return False
    
    def _recover_graceful_degradation(self, failure_event: FailureEvent) -> bool:
        """Recover by degrading performance gracefully."""
        
        try:
            logger.info("Attempting graceful degradation...")
            
            # Disable advanced features
            if hasattr(self.model, 'performance_optimizer'):
                self.model.performance_optimizer = None
                logger.info("Disabled performance optimizer")
            
            if hasattr(self.model, 'nas_optimizer'):
                self.model.nas_optimizer = None
                logger.info("Disabled NAS optimizer")
            
            # Reduce model complexity
            if hasattr(self.config, 'gradient_checkpointing'):
                self.config.gradient_checkpointing = False
            
            # Enable conservative settings
            if hasattr(self.config, 'mixed_precision'):
                self.config.mixed_precision = False
            
            if hasattr(self.config, 'batch_size'):
                self.config.batch_size = min(self.config.batch_size, 8)
            
            logger.info("Graceful degradation applied")
            return True
            
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False
    
    def _create_checkpoint(self, checkpoint_type: str):
        """Create a system checkpoint."""
        
        try:
            timestamp = int(time.time())
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_type}_{timestamp}.pt"
            
            # Prepare checkpoint data
            checkpoint_data = {
                "timestamp": timestamp,
                "checkpoint_type": checkpoint_type,
                "model_state_dict": self.model.state_dict(),
                "config": self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            }
            
            # Add model-specific state
            if hasattr(self.model, 'task_router'):
                checkpoint_data["task_router_mappings"] = {
                    "task_id_to_index": self.model.task_router.task_id_to_index,
                    "index_to_task_id": self.model.task_router.index_to_task_id,
                    "num_tasks": self.model.task_router.num_tasks
                }
            
            if hasattr(self.model, 'task_performance'):
                checkpoint_data["task_performance"] = self.model.task_performance
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Cleanup old checkpoints (keep last 10)
            checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"), 
                                    key=lambda p: p.stat().st_mtime)
            if len(checkpoint_files) > 10:
                for old_checkpoint in checkpoint_files[:-10]:
                    old_checkpoint.unlink()
            
            logger.info(f"Created {checkpoint_type} checkpoint: {checkpoint_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
    
    def _is_circuit_breaker_open(self, component: str) -> bool:
        """Check if circuit breaker is open for a component."""
        
        state = self.circuit_breaker_state[component]
        current_time = time.time()
        
        if state["state"] == "open":
            # Check if recovery timeout has passed
            if current_time - state["last_failure"] > self.circuit_breaker_thresholds["recovery_timeout"]:
                state["state"] = "half_open"
                logger.info(f"Circuit breaker for {component} moved to half-open")
                return False
            return True
        
        return False
    
    def _record_circuit_breaker_failure(self, component: str):
        """Record a failure for circuit breaker tracking."""
        
        state = self.circuit_breaker_state[component]
        current_time = time.time()
        
        # Reset counter if outside time window
        if current_time - state["last_failure"] > self.circuit_breaker_thresholds["time_window"]:
            state["failures"] = 0
        
        state["failures"] += 1
        state["last_failure"] = current_time
        
        # Trip circuit breaker if threshold exceeded
        if state["failures"] >= self.circuit_breaker_thresholds["failure_count"]:
            state["state"] = "open"
            logger.warning(f"Circuit breaker TRIPPED for {component} after {state['failures']} failures")
    
    def _update_circuit_breakers(self):
        """Update circuit breaker states."""
        
        current_time = time.time()
        
        for component, state in self.circuit_breaker_state.items():
            if state["state"] == "open":
                if current_time - state["last_failure"] > self.circuit_breaker_thresholds["recovery_timeout"]:
                    state["state"] = "half_open"
                    logger.info(f"Circuit breaker for {component} moved to half-open")
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of system failures and recovery."""
        
        if not self.failure_history:
            return {"message": "No failures recorded"}
        
        # Analyze failure patterns
        failure_counts = defaultdict(int)
        recovery_success_rate = defaultdict(list)
        recent_failures = []
        
        cutoff_time = time.time() - 3600  # Last hour
        
        for failure in self.failure_history:
            failure_counts[failure.failure_type.value] += 1
            
            if failure.recovery_attempted:
                recovery_success_rate[failure.failure_type.value].append(failure.recovery_successful)
            
            if failure.timestamp > cutoff_time:
                recent_failures.append(asdict(failure))
        
        # Calculate success rates
        success_rates = {}
        for failure_type, attempts in recovery_success_rate.items():
            if attempts:
                success_rates[failure_type] = sum(attempts) / len(attempts)
        
        return {
            "total_failures": len(self.failure_history),
            "failure_counts": dict(failure_counts),
            "recovery_success_rates": success_rates,
            "recent_failures_count": len(recent_failures),
            "recent_failures": recent_failures,
            "circuit_breaker_states": {
                component: state["state"] 
                for component, state in self.circuit_breaker_state.items()
            },
            "recovery_attempts": dict(self.recovery_attempts)
        }
    
    def cleanup(self):
        """Clean up failsafe system resources."""
        
        self.stop_monitoring()
        self.failure_history.clear()
        self.recovery_attempts.clear()
        self.last_recovery_time.clear()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock model and config for testing
    class MockModel:
        def state_dict(self):
            return {"param1": torch.tensor([1, 2, 3])}
        
        def load_state_dict(self, state_dict):
            pass
        
        def eval(self):
            pass
        
        def cpu(self):
            pass
    
    class MockConfig:
        device = torch.device("cpu")
        output_dir = "./test_output"
    
    # Test failsafe system
    failsafe = FailsafeSystem(MockModel(), MockConfig())
    failsafe.start_monitoring()
    
    # Simulate a failure
    try:
        raise RuntimeError("Out of memory error")
    except Exception as e:
        success, message = failsafe.handle_failure(
            e, 
            {"batch_size": 32, "task_id": "test"}, 
            "model"
        )
        print(f"Recovery result: {success} - {message}")
    
    # Get failure summary
    summary = failsafe.get_failure_summary()
    print(f"Failure summary: {summary}")
    
    # Cleanup
    failsafe.cleanup()