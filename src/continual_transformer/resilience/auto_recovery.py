"""
Automatic recovery system for continual learning failures.
Implements intelligent retry strategies and graceful degradation.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import logging
import torch

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback" 
    DEGRADE = "degrade"
    RESET = "reset"


@dataclass
class RecoveryAction:
    """Recovery action configuration."""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    delay: float = 1.0
    exponential_backoff: bool = True
    fallback_func: Optional[Callable] = None
    degradation_level: Optional[str] = None


class AutoRecoverySystem:
    """Automatic recovery system for ML operations."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.recovery_history = []
        self.active_degradations = set()
        self._lock = threading.Lock()
        
        # Define recovery strategies for different error types
        self.recovery_strategies = {
            RuntimeError: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                delay=1.0
            ),
            torch.cuda.OutOfMemoryError: RecoveryAction(
                strategy=RecoveryStrategy.DEGRADE,
                degradation_level="reduce_batch_size"
            ),
            ValueError: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                fallback_func=self._safe_fallback
            ),
            KeyError: RecoveryAction(
                strategy=RecoveryStrategy.RESET,
                max_attempts=1
            )
        }
        
        logger.info("Auto-recovery system initialized")
    
    def recover_from_error(
        self, 
        error: Exception, 
        operation: str, 
        context: Dict[str, Any]
    ) -> tuple[bool, Any]:
        """Attempt to recover from an error."""
        
        error_type = type(error)
        strategy = self.recovery_strategies.get(error_type)
        
        if not strategy:
            logger.warning(f"No recovery strategy for {error_type.__name__}")
            return False, None
        
        with self._lock:
            recovery_id = len(self.recovery_history)
            self.recovery_history.append({
                "id": recovery_id,
                "timestamp": time.time(),
                "error_type": error_type.__name__,
                "error_message": str(error),
                "operation": operation,
                "strategy": strategy.strategy.value,
                "context": context.copy()
            })
        
        logger.info(
            f"Attempting recovery #{recovery_id} for {error_type.__name__} "
            f"using {strategy.strategy.value} strategy"
        )
        
        try:
            if strategy.strategy == RecoveryStrategy.RETRY:
                return self._retry_recovery(error, operation, context, strategy)
            elif strategy.strategy == RecoveryStrategy.FALLBACK:
                return self._fallback_recovery(error, operation, context, strategy)
            elif strategy.strategy == RecoveryStrategy.DEGRADE:
                return self._degradation_recovery(error, operation, context, strategy)
            elif strategy.strategy == RecoveryStrategy.RESET:
                return self._reset_recovery(error, operation, context, strategy)
            else:
                return False, None
                
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            return False, None
    
    def _retry_recovery(
        self, 
        error: Exception, 
        operation: str, 
        context: Dict[str, Any],
        strategy: RecoveryAction
    ) -> tuple[bool, Any]:
        """Implement retry recovery strategy."""
        
        for attempt in range(strategy.max_attempts):
            try:
                # Calculate delay with exponential backoff
                if strategy.exponential_backoff:
                    delay = strategy.delay * (2 ** attempt)
                else:
                    delay = strategy.delay
                
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{strategy.max_attempts} after {delay}s")
                    time.sleep(delay)
                
                # Attempt to re-execute the operation
                result = self._re_execute_operation(operation, context)
                
                logger.info(f"Recovery successful after {attempt + 1} attempts")
                return True, result
                
            except Exception as retry_error:
                logger.warning(f"Retry attempt {attempt + 1} failed: {retry_error}")
                if attempt == strategy.max_attempts - 1:
                    logger.error("All retry attempts exhausted")
        
        return False, None
    
    def _fallback_recovery(
        self,
        error: Exception,
        operation: str,
        context: Dict[str, Any],
        strategy: RecoveryAction
    ) -> tuple[bool, Any]:
        """Implement fallback recovery strategy."""
        
        if strategy.fallback_func:
            try:
                result = strategy.fallback_func(error, operation, context)
                logger.info("Fallback recovery successful")
                return True, result
            except Exception as fallback_error:
                logger.error(f"Fallback recovery failed: {fallback_error}")
        
        return False, None
    
    def _degradation_recovery(
        self,
        error: Exception,
        operation: str,
        context: Dict[str, Any],
        strategy: RecoveryAction
    ) -> tuple[bool, Any]:
        """Implement graceful degradation recovery."""
        
        degradation = strategy.degradation_level
        
        if degradation == "reduce_batch_size":
            return self._reduce_batch_size_recovery(error, operation, context)
        elif degradation == "simplify_model":
            return self._simplify_model_recovery(error, operation, context)
        elif degradation == "disable_features":
            return self._disable_features_recovery(error, operation, context)
        
        return False, None
    
    def _reduce_batch_size_recovery(
        self,
        error: Exception,
        operation: str,
        context: Dict[str, Any]
    ) -> tuple[bool, Any]:
        """Reduce batch size to recover from memory errors."""
        
        if "batch_size" not in context or context["batch_size"] <= 1:
            return False, None
        
        # Reduce batch size by half
        new_batch_size = max(1, context["batch_size"] // 2)
        context["batch_size"] = new_batch_size
        
        self.active_degradations.add("reduced_batch_size")
        
        try:
            result = self._re_execute_operation(operation, context)
            logger.info(f"Recovery successful with reduced batch size: {new_batch_size}")
            return True, result
        except Exception as degrade_error:
            logger.error(f"Degradation recovery failed: {degrade_error}")
            return False, None
    
    def _simplify_model_recovery(
        self,
        error: Exception,
        operation: str,
        context: Dict[str, Any]
    ) -> tuple[bool, Any]:
        """Simplify model to recover from complexity errors."""
        
        # Disable complex features temporarily
        original_settings = {}
        
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # Disable advanced optimization
            if hasattr(config, 'enable_nas'):
                original_settings['enable_nas'] = config.enable_nas
                config.enable_nas = False
            
            # Disable knowledge distillation
            if hasattr(config, 'use_knowledge_distillation'):
                original_settings['use_knowledge_distillation'] = config.use_knowledge_distillation
                config.use_knowledge_distillation = False
        
        self.active_degradations.add("simplified_model")
        
        try:
            result = self._re_execute_operation(operation, context)
            logger.info("Recovery successful with simplified model")
            return True, result
        except Exception as simplify_error:
            # Restore original settings
            if hasattr(self.model, 'config'):
                config = self.model.config
                for key, value in original_settings.items():
                    setattr(config, key, value)
            
            logger.error(f"Model simplification recovery failed: {simplify_error}")
            return False, None
    
    def _disable_features_recovery(
        self,
        error: Exception,
        operation: str,
        context: Dict[str, Any]
    ) -> tuple[bool, Any]:
        """Disable non-essential features for recovery."""
        
        # Disable monitoring and logging temporarily
        original_monitoring = getattr(self.model, 'system_monitor', None)
        if original_monitoring:
            self.model.system_monitor = None
        
        self.active_degradations.add("disabled_features")
        
        try:
            result = self._re_execute_operation(operation, context)
            logger.info("Recovery successful with disabled features")
            return True, result
        except Exception as disable_error:
            # Restore monitoring
            if original_monitoring:
                self.model.system_monitor = original_monitoring
            
            logger.error(f"Feature disabling recovery failed: {disable_error}")
            return False, None
    
    def _reset_recovery(
        self,
        error: Exception,
        operation: str,
        context: Dict[str, Any],
        strategy: RecoveryAction
    ) -> tuple[bool, Any]:
        """Reset components to recover from state errors."""
        
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reset task router if applicable
            if hasattr(self.model, 'task_router') and hasattr(self.model.task_router, 'reset'):
                self.model.task_router.reset()
            
            # Clear any cached states
            if hasattr(self.model, 'previous_model_state'):
                self.model.previous_model_state = None
            
            result = self._re_execute_operation(operation, context)
            logger.info("Recovery successful after component reset")
            return True, result
            
        except Exception as reset_error:
            logger.error(f"Reset recovery failed: {reset_error}")
            return False, None
    
    def _re_execute_operation(self, operation: str, context: Dict[str, Any]) -> Any:
        """Re-execute the failed operation with updated context."""
        
        if operation == "forward":
            return self.model.forward(
                input_ids=context.get("input_ids"),
                attention_mask=context.get("attention_mask"),
                task_id=context.get("task_id"),
                labels=context.get("labels")
            )
        elif operation == "training_step":
            # Implementation depends on training loop structure
            pass
        elif operation == "inference":
            return self.model.predict(
                text=context.get("text"),
                task_id=context.get("task_id")
            )
        
        raise NotImplementedError(f"Re-execution not implemented for operation: {operation}")
    
    def _safe_fallback(self, error: Exception, operation: str, context: Dict[str, Any]) -> Any:
        """Safe fallback implementation."""
        
        if operation == "forward":
            # Return dummy outputs for graceful degradation
            batch_size = context.get("batch_size", 1)
            device = context.get("device", "cpu")
            
            return {
                "logits": torch.zeros((batch_size, 2), device=device),
                "hidden_states": torch.zeros((batch_size, 10, 768), device=device),
                "task_probs": torch.ones((batch_size, 1), device=device),
                "predicted_task_indices": torch.zeros((batch_size,), dtype=torch.long, device=device)
            }
        
        return None
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        
        with self._lock:
            total_recoveries = len(self.recovery_history)
            if total_recoveries == 0:
                return {"total_recoveries": 0}
            
            # Analyze recovery patterns
            strategy_counts = {}
            error_counts = {}
            
            for recovery in self.recovery_history:
                strategy = recovery["strategy"]
                error_type = recovery["error_type"]
                
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            return {
                "total_recoveries": total_recoveries,
                "strategy_distribution": strategy_counts,
                "error_distribution": error_counts,
                "active_degradations": list(self.active_degradations),
                "recovery_rate": len([r for r in self.recovery_history if r.get("success", False)]) / total_recoveries
            }
    
    def clear_degradations(self):
        """Clear all active degradations and restore normal operation."""
        
        self.active_degradations.clear()
        
        # Restore model settings
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # Re-enable features if they were disabled
            if hasattr(config, 'enable_nas'):
                config.enable_nas = True
            if hasattr(config, 'use_knowledge_distillation'):
                config.use_knowledge_distillation = True
        
        logger.info("All degradations cleared, normal operation restored")