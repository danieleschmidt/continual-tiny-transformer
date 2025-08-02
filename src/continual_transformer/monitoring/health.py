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