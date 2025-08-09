"""Resilience and reliability modules for continual learning."""

import torch
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import time
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern implementation for model resilience."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Union[Exception, tuple] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker: Attempting reset (HALF_OPEN)")
                else:
                    raise RuntimeError("Circuit breaker is OPEN - calls are blocked")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure(e)
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker: Reset to CLOSED state")
        self.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker: Opened due to {self.failure_count} failures. "
                f"Last error: {str(exception)}"
            )
        else:
            logger.debug(f"Circuit breaker: Failure {self.failure_count}/{self.failure_threshold}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "threshold": self.failure_threshold
        }

class RetryStrategy:
    """Configurable retry strategy with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to functions."""
        def wrapper(*args, **kwargs):
            return self._execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def _execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Function succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts:
                    logger.error(f"Function failed after {self.max_attempts} attempts: {str(e)}")
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + 0.5 * random.random())  # Add 0-50% jitter
        
        return delay

class GracefulDegradation:
    """Graceful degradation handler for model failures."""
    
    def __init__(self):
        self.fallback_handlers = {}
        self.degradation_modes = {
            "simple": self._simple_fallback,
            "cached": self._cached_fallback,
            "default": self._default_fallback
        }
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_handlers[operation] = fallback_func
        logger.info(f"Registered fallback for operation: {operation}")
    
    def execute_with_fallback(
        self, 
        operation: str, 
        primary_func: Callable,
        *args,
        **kwargs
    ):
        """Execute function with fallback on failure."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary operation '{operation}' failed: {str(e)}")
            return self._handle_fallback(operation, *args, **kwargs)
    
    def _handle_fallback(self, operation: str, *args, **kwargs):
        """Handle fallback execution."""
        if operation in self.fallback_handlers:
            try:
                logger.info(f"Executing fallback for operation: {operation}")
                return self.fallback_handlers[operation](*args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback for '{operation}' also failed: {str(e)}")
        
        # Use default degradation mode
        return self._default_fallback(operation, *args, **kwargs)
    
    def _simple_fallback(self, *args, **kwargs):
        """Simple fallback that returns None or empty result."""
        logger.info("Using simple fallback (None result)")
        return None
    
    def _cached_fallback(self, *args, **kwargs):
        """Fallback using cached results if available."""
        logger.info("Using cached fallback")
        # Placeholder - would implement actual caching logic
        return {"cached": True, "result": None}
    
    def _default_fallback(self, operation: str, *args, **kwargs):
        """Default fallback strategy."""
        logger.warning(f"No specific fallback for '{operation}', using default")
        return {
            "status": "degraded",
            "operation": operation,
            "message": "Service temporarily unavailable",
            "fallback": True
        }

class HealthMonitor:
    """Comprehensive health monitoring for continual learning models."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_history = []
        self.monitoring_active = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def register_health_check(self, name: str, check_func: Callable, critical: bool = False):
        """Register a health check function."""
        self.health_checks[name] = {
            "func": check_func,
            "critical": critical,
            "last_result": None,
            "last_check_time": None
        }
        logger.info(f"Registered health check: {name} (critical: {critical})")
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped health monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.check_interval):
            try:
                health_status = self.check_health()
                self._record_health_status(health_status)
                
                # Alert on critical failures
                critical_failures = [
                    check for check, result in health_status["checks"].items()
                    if not result["passed"] and self.health_checks[check]["critical"]
                ]
                
                if critical_failures:
                    logger.error(f"Critical health check failures: {critical_failures}")
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    def check_health(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        overall_status = "healthy"
        check_results = {}
        
        for name, check_info in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_info["func"]()
                end_time = time.time()
                
                if not isinstance(result, dict):
                    result = {"passed": bool(result), "message": str(result)}
                
                result["duration_ms"] = (end_time - start_time) * 1000
                
                check_results[name] = result
                check_info["last_result"] = result
                check_info["last_check_time"] = time.time()
                
                if not result.get("passed", False):
                    if check_info["critical"]:
                        overall_status = "critical"
                    elif overall_status == "healthy":
                        overall_status = "degraded"
                
            except Exception as e:
                error_result = {
                    "passed": False,
                    "message": f"Health check failed: {str(e)}",
                    "error": True
                }
                check_results[name] = error_result
                check_info["last_result"] = error_result
                
                if check_info["critical"]:
                    overall_status = "critical"
                elif overall_status == "healthy":
                    overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": check_results
        }
    
    def _record_health_status(self, status: Dict[str, Any]):
        """Record health status in history."""
        self.health_history.append(status)
        
        # Keep only last 100 records
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary."""
        if not self.health_history:
            return {"status": "no_data"}
        
        recent_statuses = [record["status"] for record in self.health_history[-10:]]
        
        return {
            "current_status": self.health_history[-1]["status"],
            "monitoring_active": self.monitoring_active,
            "total_checks": len(self.health_checks),
            "critical_checks": sum(1 for c in self.health_checks.values() if c["critical"]),
            "recent_trend": recent_statuses,
            "last_check_time": self.health_history[-1]["timestamp"]
        }

class ResilienceManager:
    """Main manager for all resilience features."""
    
    def __init__(self, model, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        
        # Initialize resilience components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get("circuit_breaker_threshold", 5),
            recovery_timeout=self.config.get("circuit_breaker_timeout", 60.0)
        )
        
        self.retry_strategy = RetryStrategy(
            max_attempts=self.config.get("max_retry_attempts", 3),
            base_delay=self.config.get("retry_base_delay", 1.0)
        )
        
        self.graceful_degradation = GracefulDegradation()
        self.health_monitor = HealthMonitor(
            check_interval=self.config.get("health_check_interval", 30.0)
        )
        
        self._setup_default_health_checks()
        self._setup_default_fallbacks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks for the model."""
        
        def check_model_loaded():
            """Check if model is properly loaded."""
            try:
                return {
                    "passed": hasattr(self.model, "forward"),
                    "message": "Model forward method available"
                }
            except Exception as e:
                return {"passed": False, "message": f"Model check failed: {e}"}
        
        def check_device_available():
            """Check if model device is available."""
            try:
                if hasattr(self.model, "parameters"):
                    device = next(self.model.parameters()).device
                    if device.type == "cuda" and not torch.cuda.is_available():
                        return {"passed": False, "message": "CUDA required but not available"}
                    return {"passed": True, "message": f"Device {device} available"}
                return {"passed": True, "message": "No device check needed"}
            except Exception as e:
                return {"passed": False, "message": f"Device check failed: {e}"}
        
        def check_memory_usage():
            """Check memory usage."""
            try:
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    usage_ratio = memory_used / memory_total
                    
                    if usage_ratio > 0.9:
                        return {
                            "passed": False,
                            "message": f"High GPU memory usage: {usage_ratio:.2%}"
                        }
                    
                    return {
                        "passed": True,
                        "message": f"GPU memory usage: {usage_ratio:.2%}"
                    }
                else:
                    return {"passed": True, "message": "CPU mode - no GPU memory check"}
            except Exception as e:
                return {"passed": False, "message": f"Memory check failed: {e}"}
        
        # Register health checks
        self.health_monitor.register_health_check("model_loaded", check_model_loaded, critical=True)
        self.health_monitor.register_health_check("device_available", check_device_available, critical=True)
        self.health_monitor.register_health_check("memory_usage", check_memory_usage, critical=False)
    
    def _setup_default_fallbacks(self):
        """Setup default fallback handlers."""
        
        def prediction_fallback(*args, **kwargs):
            """Fallback for prediction failures."""
            return {
                "predictions": [0],  # Default prediction
                "probabilities": [[1.0]],  # Default probability
                "fallback": True,
                "message": "Prediction service temporarily unavailable"
            }
        
        def training_fallback(*args, **kwargs):
            """Fallback for training failures."""
            return {
                "status": "failed",
                "fallback": True,
                "message": "Training temporarily unavailable"
            }
        
        # Register fallbacks
        self.graceful_degradation.register_fallback("predict", prediction_fallback)
        self.graceful_degradation.register_fallback("train", training_fallback)
    
    @contextmanager
    def resilient_operation(self, operation_name: str):
        """Context manager for resilient operations."""
        logger.info(f"Starting resilient operation: {operation_name}")
        start_time = time.time()
        
        try:
            yield self
        except Exception as e:
            logger.error(f"Resilient operation '{operation_name}' failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            logger.info(f"Resilient operation '{operation_name}' completed in {duration:.2f}s")
    
    def wrap_with_resilience(self, func: Callable, operation_name: str) -> Callable:
        """Wrap a function with all resilience features."""
        
        @self.circuit_breaker
        @self.retry_strategy
        def resilient_wrapper(*args, **kwargs):
            return self.graceful_degradation.execute_with_fallback(
                operation_name, func, *args, **kwargs
            )
        
        return resilient_wrapper
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        self.health_monitor.start_monitoring()
        logger.info("Resilience monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.health_monitor.stop_monitoring()
        logger.info("Resilience monitoring stopped")
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        return {
            "circuit_breaker": self.circuit_breaker.get_state(),
            "health_monitor": self.health_monitor.get_health_summary(),
            "fallback_handlers": list(self.graceful_degradation.fallback_handlers.keys()),
            "monitoring_active": self.health_monitor.monitoring_active
        }

__all__ = [
    "CircuitBreaker",
    "RetryStrategy", 
    "GracefulDegradation",
    "HealthMonitor",
    "ResilienceManager"
]