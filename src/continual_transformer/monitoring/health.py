"""Health check system for continual transformer."""

import time
import torch
import psutil
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_check = {}
        self.check_history = []
        self.max_history = 100
        
    def check_system_health(self) -> HealthCheckResult:
        """Check overall system health."""
        start_time = time.time()
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            
            # Check disk space
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > 90:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 85:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {memory.percent}%")
                
            if disk.percent > 90:
                status = HealthStatus.DEGRADED
                issues.append(f"Low disk space: {disk.percent}% used")
            
            if cpu_percent > 95 or memory.percent > 95:
                status = HealthStatus.UNHEALTHY
            
            message = "System healthy" if not issues else "; ".join(issues)
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "uptime_seconds": time.time() - self.start_time
            }
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details=details
            )
            
            self._record_check("system", result)
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"System health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
            self._record_check("system", result)
            return result
    
    def check_gpu_health(self) -> HealthCheckResult:
        """Check GPU health and availability."""
        start_time = time.time()
        
        try:
            if not torch.cuda.is_available():
                duration_ms = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="CUDA not available",
                    timestamp=datetime.now(),
                    duration_ms=duration_ms,
                    details={"cuda_available": False}
                )
            
            device_count = torch.cuda.device_count()
            gpu_details = {}
            
            for i in range(device_count):
                device = torch.device(f'cuda:{i}')
                torch.cuda.set_device(device)
                
                # Get GPU memory info
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory
                
                memory_usage_percent = (memory_allocated / memory_total) * 100
                
                gpu_details[f"gpu_{i}"] = {
                    "name": torch.cuda.get_device_name(device),
                    "memory_allocated_mb": memory_allocated / (1024**2),
                    "memory_reserved_mb": memory_reserved / (1024**2),
                    "memory_total_mb": memory_total / (1024**2),
                    "memory_usage_percent": memory_usage_percent
                }
            
            # Determine status
            max_usage = max(gpu["memory_usage_percent"] for gpu in gpu_details.values())
            
            if max_usage > 90:
                status = HealthStatus.DEGRADED
                message = f"High GPU memory usage: {max_usage:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "GPU healthy"
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details={
                    "cuda_available": True,
                    "device_count": device_count,
                    "gpus": gpu_details
                }
            )
            
            self._record_check("gpu", result)
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"GPU health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
            self._record_check("gpu", result)
            return result
    
    def check_model_health(self, model=None) -> HealthCheckResult:
        """Check model-specific health."""
        start_time = time.time()
        
        try:
            if model is None:
                duration_ms = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="No model provided for health check",
                    timestamp=datetime.now(),
                    duration_ms=duration_ms
                )
            
            # Check if model is in training mode appropriately
            is_training = model.training
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Check for NaN/Inf in parameters
            has_nan = False
            has_inf = False
            
            for param in model.parameters():
                if torch.isnan(param).any():
                    has_nan = True
                if torch.isinf(param).any():
                    has_inf = True
            
            # Determine status
            if has_nan or has_inf:
                status = HealthStatus.UNHEALTHY
                message = "Model has NaN or Inf parameters"
            else:
                status = HealthStatus.HEALTHY
                message = "Model parameters healthy"
            
            details = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "is_training": is_training,
                "has_nan_parameters": has_nan,
                "has_inf_parameters": has_inf
            }
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details=details
            )
            
            self._record_check("model", result)
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Model health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
            self._record_check("model", result)
            return result
    
    def check_dependencies(self) -> HealthCheckResult:
        """Check external dependencies health."""
        start_time = time.time()
        
        try:
            import torch
            import numpy as np
            import transformers
            
            # Version checks
            versions = {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "transformers": transformers.__version__
            }
            
            # Test basic operations
            test_tensor = torch.randn(10, 10)
            test_result = torch.matmul(test_tensor, test_tensor.T)
            
            if torch.isnan(test_result).any():
                status = HealthStatus.UNHEALTHY
                message = "Basic tensor operations producing NaN"
            else:
                status = HealthStatus.HEALTHY
                message = "Dependencies healthy"
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details={"versions": versions}
            )
            
            self._record_check("dependencies", result)
            return result
            
        except ImportError as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Missing dependency: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
            self._record_check("dependencies", result)
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
            self._record_check("dependencies", result)
            return result
    
    def comprehensive_health_check(self, model=None) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        checks = {
            "system": self.check_system_health(),
            "gpu": self.check_gpu_health(),
            "dependencies": self.check_dependencies()
        }
        
        if model is not None:
            checks["model"] = self.check_model_health(model)
        
        return checks
    
    def get_overall_status(self, checks: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall status from individual checks."""
        if any(check.status == HealthStatus.UNHEALTHY for check in checks.values()):
            return HealthStatus.UNHEALTHY
        elif any(check.status == HealthStatus.DEGRADED for check in checks.values()):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _record_check(self, check_type: str, result: HealthCheckResult):
        """Record health check result."""
        self.last_check[check_type] = result
        self.check_history.append({
            "type": check_type,
            "result": result,
            "timestamp": result.timestamp
        })
        
        # Maintain history size
        if len(self.check_history) > self.max_history:
            self.check_history.pop(0)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of health status."""
        recent_checks = {}
        for check_type, result in self.last_check.items():
            recent_checks[check_type] = {
                "status": result.status.value,
                "message": result.message,
                "last_check": result.timestamp.isoformat(),
                "duration_ms": result.duration_ms
            }
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "last_checks": recent_checks,
            "total_checks_run": len(self.check_history)
        }


class PeriodicHealthChecker:
    """Runs health checks periodically in background."""
    
    def __init__(self, health_checker: HealthChecker, interval_seconds: int = 60):
        self.health_checker = health_checker
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread = None
        
    def start(self):
        """Start periodic health checking."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_checks, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop periodic health checking."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _run_checks(self):
        """Run health checks periodically."""
        while self.running:
            try:
                self.health_checker.comprehensive_health_check()
            except Exception as e:
                print(f"Health check error: {e}")
            
            time.sleep(self.interval_seconds)


class RobustHealthMonitor:
    """Enhanced health monitor with advanced error handling and recovery."""
    
    def __init__(self, config=None):
        self.config = config
        self.health_checker = HealthChecker()
        self.periodic_checker = None
        
        # Error tracking
        self.error_history = []
        self.max_error_history = 100
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Performance metrics
        self.performance_stats = {
            "task_durations": [],
            "memory_peaks": [],
            "error_counts": {},
            "recovery_times": []
        }
        
        # Circuit breaker for health checks
        self.health_check_disabled = False
        self.last_health_check_failure = None
        self.health_check_failure_threshold = 3
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start comprehensive monitoring."""
        try:
            if self.periodic_checker is None:
                self.periodic_checker = PeriodicHealthChecker(
                    self.health_checker, 
                    interval_seconds
                )
            self.periodic_checker.start()
            print(f"✅ Health monitoring started (interval: {interval_seconds}s)")
        except Exception as e:
            self._record_error("monitoring_start_failed", str(e))
            print(f"❌ Failed to start health monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring safely."""
        try:
            if self.periodic_checker:
                self.periodic_checker.stop()
            print("✅ Health monitoring stopped")
        except Exception as e:
            self._record_error("monitoring_stop_failed", str(e))
            print(f"⚠️  Error stopping monitoring: {e}")
    
    def safe_health_check(self, model=None) -> Dict[str, Any]:
        """Perform health check with error handling and recovery."""
        if self.health_check_disabled:
            return {
                "status": "DISABLED",
                "message": "Health checks disabled due to consecutive failures",
                "last_failure": self.last_health_check_failure
            }
        
        try:
            checks = self.health_checker.comprehensive_health_check(model)
            overall_status = self.health_checker.get_overall_status(checks)
            
            # Reset failure count on success
            self.consecutive_failures = 0
            
            return {
                "status": overall_status.value.upper(),
                "checks": {
                    name: {
                        "status": check.status.value,
                        "message": check.message,
                        "duration_ms": check.duration_ms,
                        "details": check.details
                    }
                    for name, check in checks.items()
                },
                "summary": self.health_checker.get_health_summary()
            }
            
        except Exception as e:
            self.consecutive_failures += 1
            self.last_health_check_failure = str(e)
            
            error_msg = f"Health check failed: {e}"
            self._record_error("health_check_failed", error_msg)
            
            # Disable health checks if too many consecutive failures
            if self.consecutive_failures >= self.health_check_failure_threshold:
                self.health_check_disabled = True
                print(f"❌ Disabling health checks after {self.consecutive_failures} failures")
            
            return {
                "status": "ERROR",
                "message": error_msg,
                "consecutive_failures": self.consecutive_failures
            }
    
    def record_task_performance(self, task_id: str, duration_seconds: float, success: bool = True):
        """Record task performance metrics."""
        try:
            self.performance_stats["task_durations"].append({
                "task_id": task_id,
                "duration": duration_seconds,
                "success": success,
                "timestamp": datetime.now()
            })
            
            # Track memory peak during task
            try:
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    self.performance_stats["memory_peaks"].append(peak_memory)
            except Exception:
                pass  # Non-critical error
                
            # Maintain reasonable history size
            if len(self.performance_stats["task_durations"]) > 1000:
                self.performance_stats["task_durations"] = self.performance_stats["task_durations"][-500:]
                
        except Exception as e:
            self._record_error("performance_recording_failed", str(e))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance statistics summary."""
        try:
            durations = self.performance_stats["task_durations"]
            if not durations:
                return {"message": "No performance data available"}
            
            # Calculate statistics
            successful_tasks = [d for d in durations if d["success"]]
            failed_tasks = [d for d in durations if not d["success"]]
            
            avg_duration = sum(d["duration"] for d in successful_tasks) / len(successful_tasks) if successful_tasks else 0
            max_duration = max(d["duration"] for d in durations)
            min_duration = min(d["duration"] for d in durations)
            
            success_rate = len(successful_tasks) / len(durations) * 100
            
            # Memory stats
            memory_peaks = self.performance_stats["memory_peaks"]
            avg_memory = sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0
            max_memory = max(memory_peaks) if memory_peaks else 0
            
            return {
                "total_tasks": len(durations),
                "successful_tasks": len(successful_tasks),
                "failed_tasks": len(failed_tasks),
                "success_rate_percent": success_rate,
                "duration_stats": {
                    "average_seconds": avg_duration,
                    "max_seconds": max_duration,
                    "min_seconds": min_duration
                },
                "memory_stats": {
                    "average_peak_mb": avg_memory,
                    "max_peak_mb": max_memory
                },
                "error_summary": dict(self.performance_stats["error_counts"])
            }
            
        except Exception as e:
            self._record_error("performance_summary_failed", str(e))
            return {"error": f"Failed to generate performance summary: {e}"}
    
    def _record_error(self, error_type: str, error_message: str):
        """Record error with timestamp and categorization."""
        try:
            error_record = {
                "type": error_type,
                "message": error_message,
                "timestamp": datetime.now(),
                "thread_id": threading.current_thread().ident
            }
            
            self.error_history.append(error_record)
            
            # Update error counts
            if error_type in self.performance_stats["error_counts"]:
                self.performance_stats["error_counts"][error_type] += 1
            else:
                self.performance_stats["error_counts"][error_type] = 1
            
            # Maintain error history size
            if len(self.error_history) > self.max_error_history:
                self.error_history = self.error_history[-50:]  # Keep last 50
                
        except Exception:
            # If we can't even record errors, just print to console
            print(f"CRITICAL: Failed to record error - {error_type}: {error_message}")
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error report."""
        try:
            recent_errors = self.error_history[-10:] if self.error_history else []
            
            # Group errors by type
            error_groups = {}
            for error in self.error_history:
                error_type = error["type"]
                if error_type not in error_groups:
                    error_groups[error_type] = []
                error_groups[error_type].append(error)
            
            return {
                "total_errors": len(self.error_history),
                "error_types": list(self.performance_stats["error_counts"].keys()),
                "recent_errors": [
                    {
                        "type": e["type"],
                        "message": e["message"],
                        "timestamp": e["timestamp"].isoformat()
                    }
                    for e in recent_errors
                ],
                "error_counts": dict(self.performance_stats["error_counts"]),
                "consecutive_failures": self.consecutive_failures,
                "health_checks_disabled": self.health_check_disabled
            }
            
        except Exception as e:
            return {"error": f"Failed to generate error report: {e}"}
    
    def reset_health_monitoring(self):
        """Reset health monitoring state (for recovery)."""
        try:
            self.health_check_disabled = False
            self.consecutive_failures = 0
            self.last_health_check_failure = None
            print("✅ Health monitoring reset")
        except Exception as e:
            print(f"❌ Failed to reset health monitoring: {e}")
    
    def export_diagnostics(self, filepath: str):
        """Export comprehensive diagnostics report."""
        try:
            diagnostics = {
                "generated_at": datetime.now().isoformat(),
                "health_status": self.safe_health_check(),
                "performance_summary": self.get_performance_summary(),
                "error_report": self.get_error_report(),
                "system_info": {
                    "python_version": sys.version,
                    "torch_version": torch.__version__ if 'torch' in sys.modules else "Not loaded",
                    "cuda_available": torch.cuda.is_available() if 'torch' in sys.modules else False
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)
            
            print(f"✅ Diagnostics exported to {filepath}")
            
        except Exception as e:
            error_msg = f"Failed to export diagnostics: {e}"
            self._record_error("diagnostics_export_failed", error_msg)
            print(f"❌ {error_msg}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with error tracking."""
        if exc_type is not None:
            self._record_error(exc_type.__name__, str(exc_val))
        self.stop_monitoring()


# Add necessary imports
import sys
import json