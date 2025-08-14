"""
Comprehensive health monitoring system for continual learning models.
Provides real-time monitoring, alerting, and diagnostics.
"""

import time
import threading
import logging
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health metrics snapshot."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: Optional[float]
    model_memory_mb: float
    inference_latency_ms: float
    error_rate: float
    task_accuracy: Dict[str, float]
    system_load: float
    temperature: Optional[float]


@dataclass
class Alert:
    """System alert definition."""
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    component: str
    timestamp: float
    details: Dict[str, Any]


class HealthMonitor:
    """Real-time health monitoring for continual learning systems."""
    
    def __init__(
        self,
        model,
        config,
        monitoring_interval: float = 5.0,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.model = model
        self.config = config
        self.monitoring_interval = monitoring_interval
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "gpu_memory_usage": 90.0,
            "error_rate": 0.1,
            "inference_latency_ms": 1000.0,
            "temperature": 80.0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self.total_inferences = 0
        
        # Callbacks for alerts
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        logger.info("HealthMonitor initialized")
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system and model metrics."""
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        
        # GPU metrics
        gpu_memory_usage = None
        temperature = None
        
        if torch.cuda.is_available():
            try:
                gpu_memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                
                # Try to get GPU temperature (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temperature = float(temp)
                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"Could not get GPU temperature: {e}")
                    
            except Exception as e:
                logger.debug(f"Could not get GPU metrics: {e}")
        
        # Model metrics
        model_memory_mb = 0.0
        try:
            if hasattr(self.model, 'get_memory_usage'):
                memory_stats = self.model.get_memory_usage()
                model_memory_mb = memory_stats.get('total_parameters', 0) * 4 / (1024 * 1024)  # Rough estimate
        except Exception as e:
            logger.debug(f"Could not get model memory: {e}")
        
        # Performance metrics
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0.0
        
        # Error rate
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / max(self.total_inferences, 1)
        
        # Task accuracy (placeholder - would need actual evaluation)
        task_accuracy = {}
        try:
            if hasattr(self.model, 'task_performance') and self.model.task_performance:
                for task_id, perf in self.model.task_performance.items():
                    if perf.get('eval_accuracy'):
                        task_accuracy[task_id] = perf['eval_accuracy'][-1]
        except Exception as e:
            logger.debug(f"Could not get task accuracy: {e}")
        
        return HealthMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            model_memory_mb=model_memory_mb,
            inference_latency_ms=avg_inference_time,
            error_rate=error_rate,
            task_accuracy=task_accuracy,
            system_load=system_load,
            temperature=temperature
        )
    
    def _check_alerts(self, metrics: HealthMetrics):
        """Check metrics against thresholds and generate alerts."""
        
        alerts_to_send = []
        
        # CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds.get("cpu_usage", 90):
            alerts_to_send.append(Alert(
                level="WARNING",
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                component="system",
                timestamp=metrics.timestamp,
                details={"cpu_usage": metrics.cpu_usage}
            ))
        
        # Memory usage alert
        if metrics.memory_usage > self.alert_thresholds.get("memory_usage", 85):
            alerts_to_send.append(Alert(
                level="WARNING",
                message=f"High memory usage: {metrics.memory_usage:.1f}%",
                component="system",
                timestamp=metrics.timestamp,
                details={"memory_usage": metrics.memory_usage}
            ))
        
        # GPU memory alert
        if (metrics.gpu_memory_usage is not None and 
            metrics.gpu_memory_usage > self.alert_thresholds.get("gpu_memory_usage", 90)):
            alerts_to_send.append(Alert(
                level="WARNING",
                message=f"High GPU memory usage: {metrics.gpu_memory_usage:.1f}%",
                component="gpu",
                timestamp=metrics.timestamp,
                details={"gpu_memory_usage": metrics.gpu_memory_usage}
            ))
        
        # Error rate alert
        if metrics.error_rate > self.alert_thresholds.get("error_rate", 0.1):
            alerts_to_send.append(Alert(
                level="ERROR",
                message=f"High error rate: {metrics.error_rate:.3f}",
                component="model",
                timestamp=metrics.timestamp,
                details={"error_rate": metrics.error_rate}
            ))
        
        # Inference latency alert
        if metrics.inference_latency_ms > self.alert_thresholds.get("inference_latency_ms", 1000):
            alerts_to_send.append(Alert(
                level="WARNING",
                message=f"High inference latency: {metrics.inference_latency_ms:.1f}ms",
                component="performance",
                timestamp=metrics.timestamp,
                details={"inference_latency_ms": metrics.inference_latency_ms}
            ))
        
        # Temperature alert
        if (metrics.temperature is not None and 
            metrics.temperature > self.alert_thresholds.get("temperature", 80)):
            alerts_to_send.append(Alert(
                level="CRITICAL",
                message=f"High temperature: {metrics.temperature:.1f}°C",
                component="hardware",
                timestamp=metrics.timestamp,
                details={"temperature": metrics.temperature}
            ))
        
        # Task accuracy degradation alert
        for task_id, accuracy in metrics.task_accuracy.items():
            if accuracy < 0.5:  # Threshold for accuracy degradation
                alerts_to_send.append(Alert(
                    level="WARNING",
                    message=f"Low accuracy for task {task_id}: {accuracy:.3f}",
                    component="model",
                    timestamp=metrics.timestamp,
                    details={"task_id": task_id, "accuracy": accuracy}
                ))
        
        # Send alerts
        for alert in alerts_to_send:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Alert):
        """Send alert to registered callbacks and store in history."""
        
        self.alerts.append(alert)
        
        # Log alert
        level_map = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        logger.log(level_map.get(alert.level, logging.INFO), 
                  f"[{alert.component}] {alert.message}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def record_inference(self, latency_ms: float):
        """Record inference timing."""
        self.inference_times.append(latency_ms)
        self.total_inferences += 1
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status."""
        
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics collected yet"}
        
        latest_metrics = self.metrics_history[-1]
        recent_alerts = [alert for alert in self.alerts 
                        if alert.timestamp > time.time() - 3600]  # Last hour
        
        # Determine overall health status
        critical_alerts = [a for a in recent_alerts if a.level == "CRITICAL"]
        error_alerts = [a for a in recent_alerts if a.level == "ERROR"]
        warning_alerts = [a for a in recent_alerts if a.level == "WARNING"]
        
        if critical_alerts:
            status = "critical"
        elif error_alerts:
            status = "error"
        elif warning_alerts:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": latest_metrics.timestamp,
            "metrics": asdict(latest_metrics),
            "recent_alerts": [asdict(alert) for alert in recent_alerts],
            "alert_counts": {
                "critical": len(critical_alerts),
                "error": len(error_alerts),
                "warning": len(warning_alerts)
            },
            "total_inferences": self.total_inferences,
            "total_errors": sum(self.error_counts.values())
        }
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period."""
        
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            asdict(metrics) for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        
        if not self.metrics_history:
            return {"message": "No data available"}
        
        # Calculate statistics over recent metrics
        recent_metrics = list(self.metrics_history)[-min(60, len(self.metrics_history)):]
        
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        latency_values = [m.inference_latency_ms for m in recent_metrics if m.inference_latency_ms > 0]
        
        return {
            "timespan_minutes": len(recent_metrics) * self.monitoring_interval / 60,
            "cpu_usage": {
                "min": min(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "avg": np.mean(cpu_values) if cpu_values else 0,
                "current": recent_metrics[-1].cpu_usage if recent_metrics else 0
            },
            "memory_usage": {
                "min": min(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "avg": np.mean(memory_values) if memory_values else 0,
                "current": recent_metrics[-1].memory_usage if recent_metrics else 0
            },
            "inference_latency": {
                "min": min(latency_values) if latency_values else 0,
                "max": max(latency_values) if latency_values else 0,
                "avg": np.mean(latency_values) if latency_values else 0,
                "p95": np.percentile(latency_values, 95) if latency_values else 0
            },
            "error_summary": dict(self.error_counts),
            "total_inferences": self.total_inferences,
            "uptime_minutes": (time.time() - recent_metrics[0].timestamp) / 60 if recent_metrics else 0
        }
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics history to file."""
        
        metrics_data = {
            "export_timestamp": time.time(),
            "export_hours": hours,
            "metrics": self.get_metrics_history(hours),
            "alerts": [asdict(alert) for alert in self.alerts 
                      if alert.timestamp > time.time() - (hours * 3600)],
            "summary": self.get_performance_summary(),
            "thresholds": self.alert_thresholds
        }
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")
    
    def cleanup(self):
        """Clean up monitoring resources."""
        self.stop_monitoring()
        self.metrics_history.clear()
        self.alerts.clear()
        self.alert_callbacks.clear()


class AlertNotifier:
    """Alert notification system with multiple channels."""
    
    def __init__(self):
        self.channels = {}
    
    def add_email_channel(self, smtp_config: Dict[str, Any]):
        """Add email notification channel."""
        # Placeholder for email implementation
        logger.info("Email notification channel added")
    
    def add_webhook_channel(self, webhook_url: str):
        """Add webhook notification channel."""
        # Placeholder for webhook implementation
        logger.info(f"Webhook notification channel added: {webhook_url}")
    
    def add_log_channel(self, log_file: str):
        """Add log file notification channel."""
        def log_alert(alert: Alert):
            with open(log_file, 'a') as f:
                f.write(f"{datetime.fromtimestamp(alert.timestamp)} [{alert.level}] "
                       f"{alert.component}: {alert.message}\n")
        
        self.channels['log'] = log_alert
        logger.info(f"Log notification channel added: {log_file}")
    
    def send_alert(self, alert: Alert):
        """Send alert through all configured channels."""
        for channel_name, channel_func in self.channels.items():
            try:
                channel_func(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel_name}: {e}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock model for testing
    class MockModel:
        def get_memory_usage(self):
            return {"total_parameters": 1000000}
        
        task_performance = {
            "test_task": {"eval_accuracy": [0.85]}
        }
    
    # Mock config
    class MockConfig:
        pass
    
    # Test health monitor
    monitor = HealthMonitor(MockModel(), MockConfig(), monitoring_interval=1.0)
    
    # Add alert callback
    def print_alert(alert: Alert):
        print(f"ALERT: {alert.level} - {alert.message}")
    
    monitor.add_alert_callback(print_alert)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Let it run for a few seconds
    time.sleep(5)
    
    # Check status
    status = monitor.get_current_status()
    print(f"Status: {status['status']}")
    
    # Stop monitoring
    monitor.stop_monitoring()