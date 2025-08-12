"""
Circuit breaker pattern implementation for robust continual learning.
Prevents cascade failures and provides graceful degradation.
"""

import time
import threading
from enum import Enum
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking all requests  
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics."""
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker implementation for ML model operations."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: tuple = (Exception,),
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property 
    def metrics(self) -> CircuitMetrics:
        """Get circuit metrics."""
        return self._metrics
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._metrics.last_failure_time is None:
            return True
        
        return (time.time() - self._metrics.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            self._metrics.success_count += 1
            self._metrics.total_requests += 1
            self._metrics.last_success_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._metrics.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' CLOSED after recovery")
    
    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self._metrics.failure_count += 1
            self._metrics.total_requests += 1
            self._metrics.last_failure_time = time.time()
            
            if (self._metrics.failure_count >= self.failure_threshold and 
                self._state != CircuitState.OPEN):
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPENED after {self._metrics.failure_count} failures"
                )
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._metrics.failure_count = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._metrics.failure_count,
            "success_count": self._metrics.success_count,
            "total_requests": self._metrics.total_requests,
            "failure_rate": (
                self._metrics.failure_count / max(self._metrics.total_requests, 1)
            ),
            "last_failure_time": self._metrics.last_failure_time,
            "last_success_time": self._metrics.last_success_time
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ModelCircuitBreaker:
    """Circuit breaker specifically for ML model operations."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Create circuit breakers for different operations
        self.breakers = {
            "forward": CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                recovery_timeout=config.circuit_breaker_timeout,
                expected_exception=(RuntimeError, ValueError, torch.cuda.OutOfMemoryError),
                name="forward_pass"
            ),
            "training": CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                recovery_timeout=config.circuit_breaker_timeout,
                expected_exception=(RuntimeError, ValueError),
                name="training"
            ),
            "inference": CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold // 2,
                recovery_timeout=config.circuit_breaker_timeout // 2,
                expected_exception=(RuntimeError, ValueError),
                name="inference"
            )
        }
    
    def protected_forward(self, *args, **kwargs):
        """Forward pass with circuit breaker protection."""
        return self.breakers["forward"].call(self.model.forward, *args, **kwargs)
    
    def protected_training_step(self, train_func, *args, **kwargs):
        """Training step with circuit breaker protection.""" 
        return self.breakers["training"].call(train_func, *args, **kwargs)
    
    def protected_inference(self, inference_func, *args, **kwargs):
        """Inference with circuit breaker protection."""
        return self.breakers["inference"].call(inference_func, *args, **kwargs)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
        logger.info("All model circuit breakers reset")