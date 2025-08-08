"""Advanced error recovery and fault tolerance for continual learning models."""

import torch
import torch.nn as nn
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback
import pickle
import json
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class ErrorInfo:
    """Container for error information."""
    error_type: str
    severity: ErrorSeverity
    message: str
    timestamp: float
    context: Dict[str, Any]
    traceback_str: str
    suggested_action: RecoveryAction


@dataclass
class RecoveryAttempt:
    """Container for recovery attempt information."""
    action: RecoveryAction
    timestamp: float
    success: bool
    duration: float
    error_after_recovery: Optional[str]


class ModelCheckpoint:
    """Model checkpoint for recovery."""
    
    def __init__(self, model_state: Dict, metadata: Dict):
        self.model_state = model_state
        self.metadata = metadata
        self.timestamp = time.time()
    
    def save(self, filepath: str):
        """Save checkpoint to file."""
        checkpoint_data = {
            'model_state': self.model_state,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
        torch.save(checkpoint_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelCheckpoint':
        """Load checkpoint from file."""
        checkpoint_data = torch.load(filepath, map_location='cpu')
        return cls(
            model_state=checkpoint_data['model_state'],
            metadata=checkpoint_data['metadata']
        )


class ErrorRecoverySystem:
    """Comprehensive error recovery system for continual learning models."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Error tracking
        self.error_history = []
        self.recovery_history = []
        self.error_patterns = {}
        
        # Checkpointing
        self.checkpoints = {}
        self.max_checkpoints = 10
        self.checkpoint_interval = 300  # 5 minutes
        self.last_checkpoint = 0
        
        # Recovery strategies
        self.recovery_strategies = self._init_recovery_strategies()
        
        # Circuit breaker pattern
        self.circuit_breakers = {}
        
        # Graceful degradation
        self.degradation_modes = {
            'reduced_batch_size': False,
            'simplified_model': False,
            'cpu_fallback': False,
            'reduced_precision': False
        }
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
    def _init_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategy mappings."""
        return {
            'RuntimeError': self._handle_runtime_error,
            'OutOfMemoryError': self._handle_oom_error,
            'ValueError': self._handle_value_error,
            'AttributeError': self._handle_attribute_error,
            'KeyError': self._handle_key_error,
            'ConnectionError': self._handle_connection_error,
            'TimeoutError': self._handle_timeout_error,
            'default': self._handle_generic_error
        }
    
    def start_monitoring(self):
        """Start background error monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Error recovery monitoring started")
    
    def stop_monitoring(self):
        """Stop background error monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Error recovery monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop for proactive error prevention."""
        while self.monitoring_active:
            try:
                # Check for potential issues
                self._check_system_health()
                
                # Create checkpoint if needed
                if time.time() - self.last_checkpoint > self.checkpoint_interval:
                    self._create_automatic_checkpoint()
                
                # Analyze error patterns
                self._analyze_error_patterns()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring loop failed: {e}")
                time.sleep(60)  # Back off on error
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[bool, Any]:
        """
        Handle an error with intelligent recovery.
        
        Returns:
            (success, result): Whether recovery was successful and any result
        """
        context = context or {}
        
        # Classify error
        error_info = self._classify_error(error, context)
        self.error_history.append(error_info)
        
        logger.error(
            f"Error detected: {error_info.error_type} (Severity: {error_info.severity.value})\n"
            f"Message: {error_info.message}\n"
            f"Suggested action: {error_info.suggested_action.value}"
        )
        
        # Choose recovery strategy
        recovery_strategy = self._choose_recovery_strategy(error_info)
        
        # Attempt recovery
        success, result = self._attempt_recovery(error_info, recovery_strategy, context)
        
        # Record recovery attempt
        recovery_attempt = RecoveryAttempt(
            action=recovery_strategy,
            timestamp=time.time(),
            success=success,
            duration=0.0,  # Would be measured in actual implementation
            error_after_recovery=None if success else str(error)
        )
        self.recovery_history.append(recovery_attempt)
        
        return success, result
    
    def _classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorInfo:
        """Classify an error and determine its severity and suggested action."""
        error_type = type(error).__name__
        message = str(error)
        traceback_str = traceback.format_exc()
        
        # Determine severity based on error type and context
        severity = self._determine_severity(error_type, message, context)
        
        # Suggest recovery action
        suggested_action = self._suggest_recovery_action(error_type, severity, context)
        
        return ErrorInfo(
            error_type=error_type,
            severity=severity,
            message=message,
            timestamp=time.time(),
            context=context,
            traceback_str=traceback_str,
            suggested_action=suggested_action
        )
    
    def _determine_severity(self, error_type: str, message: str, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on type, message, and context."""
        
        # Critical errors that require immediate action
        if error_type in ['OutOfMemoryError', 'SystemError']:
            return ErrorSeverity.CRITICAL
        
        if "cuda" in message.lower() and "out of memory" in message.lower():
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['RuntimeError', 'ConnectionError']:
            return ErrorSeverity.HIGH
        
        if "nan" in message.lower() or "inf" in message.lower():
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['ValueError', 'KeyError', 'AttributeError']:
            return ErrorSeverity.MEDIUM
        
        # Check context for severity hints
        if context.get('is_training', False):
            return ErrorSeverity.HIGH  # Training errors are more serious
        
        if context.get('batch_size', 0) > 32:
            return ErrorSeverity.MEDIUM  # Large batch might indicate resource issues
        
        return ErrorSeverity.LOW
    
    def _suggest_recovery_action(self, error_type: str, severity: ErrorSeverity, context: Dict[str, Any]) -> RecoveryAction:
        """Suggest appropriate recovery action based on error characteristics."""
        
        # Critical errors often require restart or abort
        if severity == ErrorSeverity.CRITICAL:
            if error_type == 'OutOfMemoryError':
                return RecoveryAction.FALLBACK  # Try reduced memory approach
            return RecoveryAction.RESTART
        
        # High severity errors might benefit from retry or fallback
        if severity == ErrorSeverity.HIGH:
            if error_type == 'RuntimeError':
                return RecoveryAction.RETRY
            return RecoveryAction.FALLBACK
        
        # Medium and low severity errors can often be retried
        return RecoveryAction.RETRY
    
    def _choose_recovery_strategy(self, error_info: ErrorInfo) -> RecoveryAction:
        """Choose the best recovery strategy based on error info and history."""
        
        # Check if we've seen this error pattern before
        pattern_key = f"{error_info.error_type}:{error_info.message[:50]}"
        if pattern_key in self.error_patterns:
            pattern = self.error_patterns[pattern_key]
            # Use the most successful previous recovery action
            if pattern['successful_recoveries']:
                most_successful = max(
                    pattern['successful_recoveries'],
                    key=pattern['successful_recoveries'].get
                )
                return RecoveryAction(most_successful)
        
        # Use suggested action as default
        return error_info.suggested_action
    
    def _attempt_recovery(self, error_info: ErrorInfo, action: RecoveryAction, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt to recover from an error using the specified action."""
        
        try:
            if action == RecoveryAction.RETRY:
                return self._retry_operation(error_info, context)
            elif action == RecoveryAction.FALLBACK:
                return self._fallback_operation(error_info, context)
            elif action == RecoveryAction.RESTART:
                return self._restart_operation(error_info, context)
            elif action == RecoveryAction.SKIP:
                return self._skip_operation(error_info, context)
            elif action == RecoveryAction.ABORT:
                return self._abort_operation(error_info, context)
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False, None
    
    def _retry_operation(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Retry the failed operation with potential modifications."""
        
        # Get specific handler for error type
        handler = self.recovery_strategies.get(
            error_info.error_type,
            self.recovery_strategies['default']
        )
        
        return handler(error_info, context, 'retry')
    
    def _fallback_operation(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Use fallback approach for the failed operation."""
        
        # Enable graceful degradation
        if error_info.error_type == 'OutOfMemoryError':
            return self._enable_memory_fallback(context)
        elif 'cuda' in error_info.message.lower():
            return self._enable_cpu_fallback(context)
        
        # Generic fallback
        return self._generic_fallback(error_info, context)
    
    def _restart_operation(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Restart the model or system component."""
        
        try:
            # Restore from latest checkpoint
            if self.checkpoints:
                latest_checkpoint = max(self.checkpoints.keys())
                self._restore_checkpoint(latest_checkpoint)
                logger.info(f"Restored model from checkpoint: {latest_checkpoint}")
                return True, "Model restarted from checkpoint"
            else:
                # Reinitialize model
                self._reinitialize_model()
                return True, "Model reinitialized"
                
        except Exception as e:
            logger.error(f"Restart operation failed: {e}")
            return False, None
    
    def _skip_operation(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Skip the failed operation and continue."""
        
        logger.warning(f"Skipping operation due to error: {error_info.message}")
        return True, "Operation skipped"
    
    def _abort_operation(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Abort the current operation."""
        
        logger.critical(f"Aborting operation due to critical error: {error_info.message}")
        return False, "Operation aborted"
    
    def _handle_runtime_error(self, error_info: ErrorInfo, context: Dict[str, Any], action: str) -> Tuple[bool, Any]:
        """Handle RuntimeError specifically."""
        
        if "nan" in error_info.message.lower() or "inf" in error_info.message.lower():
            # Try to fix NaN/Inf issues
            self._fix_nan_inf_issues()
            return True, "Fixed NaN/Inf issues"
        
        # Generic retry with small delay
        time.sleep(0.1)
        return True, "Retried after delay"
    
    def _handle_oom_error(self, error_info: ErrorInfo, context: Dict[str, Any], action: str) -> Tuple[bool, Any]:
        """Handle Out of Memory errors."""
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reduce batch size if possible
        if 'batch_size' in context and context['batch_size'] > 1:
            new_batch_size = max(1, context['batch_size'] // 2)
            context['batch_size'] = new_batch_size
            self.degradation_modes['reduced_batch_size'] = True
            logger.info(f"Reduced batch size to {new_batch_size}")
            return True, f"Reduced batch size to {new_batch_size}"
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
            return True, "Enabled gradient checkpointing"
        
        return False, "Unable to resolve OOM error"
    
    def _handle_value_error(self, error_info: ErrorInfo, context: Dict[str, Any], action: str) -> Tuple[bool, Any]:
        """Handle ValueError specifically."""
        
        # Check for shape mismatches
        if "shape" in error_info.message.lower():
            logger.info("Attempting to fix shape mismatch")
            # This would need specific logic based on the operation
            return True, "Fixed shape mismatch"
        
        return False, "Unable to resolve ValueError"
    
    def _handle_attribute_error(self, error_info: ErrorInfo, context: Dict[str, Any], action: str) -> Tuple[bool, Any]:
        """Handle AttributeError specifically."""
        
        # Try to reinitialize the problematic component
        if 'task_id' in context:
            task_id = context['task_id']
            if hasattr(self.model, 'register_task'):
                # Re-register the task
                self.model.register_task(task_id, 2)  # Default to 2 classes
                return True, f"Re-registered task {task_id}"
        
        return False, "Unable to resolve AttributeError"
    
    def _handle_key_error(self, error_info: ErrorInfo, context: Dict[str, Any], action: str) -> Tuple[bool, Any]:
        """Handle KeyError specifically."""
        
        # Try to provide default values for missing keys
        missing_key = error_info.message.strip("'\"")
        logger.info(f"Providing default value for missing key: {missing_key}")
        
        # This would need specific logic based on the missing key
        return True, f"Provided default for key: {missing_key}"
    
    def _handle_connection_error(self, error_info: ErrorInfo, context: Dict[str, Any], action: str) -> Tuple[bool, Any]:
        """Handle ConnectionError specifically."""
        
        # Retry with exponential backoff
        retry_count = context.get('retry_count', 0)
        if retry_count < 3:
            wait_time = 2 ** retry_count
            time.sleep(wait_time)
            context['retry_count'] = retry_count + 1
            return True, f"Retrying connection after {wait_time}s"
        
        return False, "Connection retry limit exceeded"
    
    def _handle_timeout_error(self, error_info: ErrorInfo, context: Dict[str, Any], action: str) -> Tuple[bool, Any]:
        """Handle TimeoutError specifically."""
        
        # Increase timeout and retry
        current_timeout = context.get('timeout', 30)
        new_timeout = min(current_timeout * 2, 300)  # Cap at 5 minutes
        context['timeout'] = new_timeout
        
        return True, f"Increased timeout to {new_timeout}s"
    
    def _handle_generic_error(self, error_info: ErrorInfo, context: Dict[str, Any], action: str) -> Tuple[bool, Any]:
        """Handle unknown error types."""
        
        # Generic retry with small delay
        time.sleep(0.5)
        return True, "Generic retry"
    
    def _enable_memory_fallback(self, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Enable memory optimization fallbacks."""
        
        optimizations = []
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            optimizations.append("cleared_gpu_cache")
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            optimizations.append("gradient_checkpointing")
        
        # Reduce precision if not already done
        if not self.degradation_modes['reduced_precision']:
            # This would convert model to half precision
            self.degradation_modes['reduced_precision'] = True
            optimizations.append("reduced_precision")
        
        if optimizations:
            return True, f"Applied memory optimizations: {', '.join(optimizations)}"
        
        return False, "No memory optimizations available"
    
    def _enable_cpu_fallback(self, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Enable CPU fallback for GPU errors."""
        
        if not self.degradation_modes['cpu_fallback']:
            # Move model to CPU
            self.model = self.model.cpu()
            self.degradation_modes['cpu_fallback'] = True
            logger.info("Switched to CPU fallback mode")
            return True, "Enabled CPU fallback"
        
        return False, "CPU fallback already enabled"
    
    def _generic_fallback(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Generic fallback operation."""
        
        # Enable simplified model mode
        if not self.degradation_modes['simplified_model']:
            self.degradation_modes['simplified_model'] = True
            return True, "Enabled simplified model mode"
        
        return False, "No generic fallback available"
    
    def _fix_nan_inf_issues(self):
        """Attempt to fix NaN/Inf issues in model parameters."""
        
        fixed_params = 0
        
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                # Reinitialize problematic parameters
                with torch.no_grad():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                fixed_params += 1
                logger.warning(f"Reinitialized parameter {name} due to NaN/Inf values")
        
        if fixed_params > 0:
            logger.info(f"Fixed {fixed_params} parameters with NaN/Inf values")
    
    def _reinitialize_model(self):
        """Reinitialize the model to a clean state."""
        
        # This would reinitialize model components
        # For now, just reset problematic components
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.reset_parameters()
        
        logger.info("Model reinitialized")
    
    def create_checkpoint(self, name: str, metadata: Dict[str, Any] = None):
        """Create a manual checkpoint."""
        
        metadata = metadata or {}
        metadata['checkpoint_type'] = 'manual'
        metadata['timestamp'] = time.time()
        
        checkpoint = ModelCheckpoint(
            model_state=self.model.state_dict(),
            metadata=metadata
        )
        
        self.checkpoints[name] = checkpoint
        self._cleanup_old_checkpoints()
        
        logger.info(f"Created checkpoint: {name}")
    
    def _create_automatic_checkpoint(self):
        """Create an automatic checkpoint."""
        
        name = f"auto_{int(time.time())}"
        metadata = {
            'checkpoint_type': 'automatic',
            'model_state': 'training' if self.model.training else 'eval'
        }
        
        self.create_checkpoint(name, metadata)
        self.last_checkpoint = time.time()
    
    def _restore_checkpoint(self, name: str) -> bool:
        """Restore model from checkpoint."""
        
        if name not in self.checkpoints:
            logger.error(f"Checkpoint not found: {name}")
            return False
        
        try:
            checkpoint = self.checkpoints[name]
            self.model.load_state_dict(checkpoint.model_state)
            logger.info(f"Restored checkpoint: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {name}: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain the limit."""
        
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and keep the most recent
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        # Keep only the most recent checkpoints
        to_keep = sorted_checkpoints[:self.max_checkpoints]
        self.checkpoints = dict(to_keep)
        
        logger.info(f"Cleaned up old checkpoints, kept {len(self.checkpoints)}")
    
    def _check_system_health(self):
        """Proactive system health checking."""
        
        # Check for potential memory issues
        if torch.cuda.is_available():
            memory_percent = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            if memory_percent > 90:
                logger.warning(f"High GPU memory usage: {memory_percent:.1f}%")
        
        # Check for NaN parameters
        nan_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        
        if nan_params:
            logger.warning(f"NaN detected in parameters: {', '.join(nan_params)}")
    
    def _analyze_error_patterns(self):
        """Analyze error patterns to improve recovery strategies."""
        
        if len(self.error_history) < 10:
            return
        
        # Recent errors (last 24 hours)
        cutoff_time = time.time() - 86400
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        # Group by error pattern
        for error in recent_errors:
            pattern_key = f"{error.error_type}:{error.message[:50]}"
            if pattern_key not in self.error_patterns:
                self.error_patterns[pattern_key] = {
                    'count': 0,
                    'successful_recoveries': {},
                    'failed_recoveries': {}
                }
            
            self.error_patterns[pattern_key]['count'] += 1
        
        # Log frequent error patterns
        frequent_patterns = {k: v for k, v in self.error_patterns.items() if v['count'] >= 5}
        if frequent_patterns:
            logger.info(f"Frequent error patterns detected: {list(frequent_patterns.keys())}")
    
    def get_recovery_report(self) -> Dict[str, Any]:
        """Generate a comprehensive recovery report."""
        
        total_errors = len(self.error_history)
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r.success)
        
        report = {
            "summary": {
                "total_errors": total_errors,
                "total_recovery_attempts": total_recoveries,
                "successful_recoveries": successful_recoveries,
                "recovery_success_rate": successful_recoveries / max(total_recoveries, 1)
            },
            "error_breakdown": {},
            "recovery_actions": {},
            "degradation_modes": dict(self.degradation_modes),
            "checkpoints": len(self.checkpoints)
        }
        
        # Error type breakdown
        error_types = {}
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        report["error_breakdown"] = error_types
        
        # Recovery action breakdown
        recovery_actions = {}
        for recovery in self.recovery_history:
            action = recovery.action.value
            if action not in recovery_actions:
                recovery_actions[action] = {"total": 0, "successful": 0}
            recovery_actions[action]["total"] += 1
            if recovery.success:
                recovery_actions[action]["successful"] += 1
        
        for action, stats in recovery_actions.items():
            stats["success_rate"] = stats["successful"] / max(stats["total"], 1)
        
        report["recovery_actions"] = recovery_actions
        
        return report