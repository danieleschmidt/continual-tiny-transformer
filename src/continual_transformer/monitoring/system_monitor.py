"""Advanced system monitoring for continual learning models."""

import torch
import psutil
import time
import threading
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_memory_used_mb: Optional[float]
    gpu_utilization: Optional[float]
    temperature: Optional[float]
    inference_time_ms: float
    throughput: float
    model_accuracy: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthStatus:
    """Container for system health status."""
    overall_health: str  # healthy, warning, critical
    components: Dict[str, str]  # component -> status
    alerts: List[str]
    recommendations: List[str]
    last_check: float
    
    def is_healthy(self) -> bool:
        return self.overall_health == "healthy"


class SystemMonitor:
    """Comprehensive system monitoring for continual learning models."""
    
    def __init__(self, model, config, monitoring_interval: float = 5.0):
        self.model = model
        self.config = config
        self.monitoring_interval = monitoring_interval
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.performance_baselines = {}
        self.alert_thresholds = self._init_alert_thresholds()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_health_check = 0
        
        # Alert system
        self.alert_callbacks = []
        self.active_alerts = set()
        
        # Performance tracking
        self.task_performance = defaultdict(list)
        self.inference_times = deque(maxlen=100)
        
    def _init_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize default alert thresholds."""
        return {
            "memory": {
                "warning": 80.0,  # 80% memory usage
                "critical": 95.0  # 95% memory usage
            },
            "gpu_memory": {
                "warning": 85.0,
                "critical": 95.0
            },
            "temperature": {
                "warning": 75.0,  # Celsius
                "critical": 85.0
            },
            "inference_time": {
                "warning": 1000.0,  # 1 second
                "critical": 5000.0  # 5 seconds
            },
            "accuracy": {
                "warning": 0.7,  # Below 70% accuracy
                "critical": 0.5   # Below 50% accuracy
            }
        }
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=self.monitoring_interval + 1)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Perform health check
                health_status = self.check_system_health(metrics)
                
                # Handle alerts
                self._process_alerts(health_status)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        timestamp = time.time()
        
        # CPU and Memory metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / 1024 / 1024
        
        # GPU metrics
        gpu_memory_used_mb = None
        gpu_utilization = None
        temperature = None
        
        if torch.cuda.is_available():
            try:
                gpu_memory_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
                
                # Try to get GPU utilization and temperature
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except ImportError:
                    logger.debug("pynvml not available for detailed GPU metrics")
                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")
            except Exception as e:
                logger.debug(f"GPU memory collection failed: {e}")
        
        # Model performance metrics
        inference_time_ms = np.mean(self.inference_times) if self.inference_times else 0.0
        throughput = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0
        
        # Model accuracy (if available)
        model_accuracy = self._get_recent_accuracy()
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_utilization=gpu_utilization,
            temperature=temperature,
            inference_time_ms=inference_time_ms,
            throughput=throughput,
            model_accuracy=model_accuracy
        )
    
    def check_system_health(self, metrics: SystemMetrics) -> HealthStatus:
        """Perform comprehensive system health check."""
        components = {}
        alerts = []
        recommendations = []
        
        # Memory health check
        if metrics.memory_percent >= self.alert_thresholds["memory"]["critical"]:
            components["memory"] = "critical"
            alerts.append(f"Critical memory usage: {metrics.memory_percent:.1f}%")
            recommendations.append("Consider reducing batch size or enabling gradient checkpointing")
        elif metrics.memory_percent >= self.alert_thresholds["memory"]["warning"]:
            components["memory"] = "warning"
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            recommendations.append("Monitor memory usage trends")
        else:
            components["memory"] = "healthy"
        
        # GPU memory health check
        if metrics.gpu_memory_used_mb is not None:
            # Get total GPU memory
            if torch.cuda.is_available():
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                gpu_memory_percent = (metrics.gpu_memory_used_mb / total_gpu_memory) * 100
                
                if gpu_memory_percent >= self.alert_thresholds["gpu_memory"]["critical"]:
                    components["gpu_memory"] = "critical"
                    alerts.append(f"Critical GPU memory usage: {gpu_memory_percent:.1f}%")
                    recommendations.append("Reduce model size or batch size")
                elif gpu_memory_percent >= self.alert_thresholds["gpu_memory"]["warning"]:
                    components["gpu_memory"] = "warning"
                    alerts.append(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
                else:
                    components["gpu_memory"] = "healthy"
        
        # Temperature health check
        if metrics.temperature is not None:
            if metrics.temperature >= self.alert_thresholds["temperature"]["critical"]:
                components["temperature"] = "critical"
                alerts.append(f"Critical temperature: {metrics.temperature}°C")
                recommendations.append("Improve cooling or reduce workload")
            elif metrics.temperature >= self.alert_thresholds["temperature"]["warning"]:
                components["temperature"] = "warning"
                alerts.append(f"High temperature: {metrics.temperature}°C")
            else:
                components["temperature"] = "healthy"
        
        # Performance health check
        if metrics.inference_time_ms >= self.alert_thresholds["inference_time"]["critical"]:
            components["performance"] = "critical"
            alerts.append(f"Critical inference time: {metrics.inference_time_ms:.1f}ms")
            recommendations.append("Optimize model or enable performance optimizations")
        elif metrics.inference_time_ms >= self.alert_thresholds["inference_time"]["warning"]:
            components["performance"] = "warning"
            alerts.append(f"Slow inference time: {metrics.inference_time_ms:.1f}ms")
        else:
            components["performance"] = "healthy"
        
        # Model accuracy health check
        if metrics.model_accuracy is not None:
            if metrics.model_accuracy <= self.alert_thresholds["accuracy"]["critical"]:
                components["accuracy"] = "critical"
                alerts.append(f"Critical accuracy drop: {metrics.model_accuracy:.2f}")
                recommendations.append("Review model training or data quality")
            elif metrics.model_accuracy <= self.alert_thresholds["accuracy"]["warning"]:
                components["accuracy"] = "warning"
                alerts.append(f"Low accuracy: {metrics.model_accuracy:.2f}")
        
        # Determine overall health
        if any(status == "critical" for status in components.values()):
            overall_health = "critical"
        elif any(status == "warning" for status in components.values()):
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        return HealthStatus(
            overall_health=overall_health,
            components=components,
            alerts=alerts,
            recommendations=recommendations,
            last_check=time.time()
        )
    
    def _process_alerts(self, health_status: HealthStatus):
        """Process and handle system alerts."""
        # Generate new alerts
        new_alerts = set(health_status.alerts) - self.active_alerts
        resolved_alerts = self.active_alerts - set(health_status.alerts)
        
        # Handle new alerts
        for alert in new_alerts:
            logger.warning(f"NEW ALERT: {alert}")
            for callback in self.alert_callbacks:
                try:
                    callback("new", alert, health_status)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
        
        # Handle resolved alerts
        for alert in resolved_alerts:
            logger.info(f"RESOLVED: {alert}")
            for callback in self.alert_callbacks:
                try:
                    callback("resolved", alert, health_status)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
        
        self.active_alerts = set(health_status.alerts)
    
    def add_alert_callback(self, callback: Callable[[str, str, HealthStatus], None]):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def record_inference_time(self, inference_time_ms: float):
        """Record inference time for performance monitoring."""
        self.inference_times.append(inference_time_ms)
    
    def record_task_performance(self, task_id: str, accuracy: float):
        """Record task performance for accuracy monitoring."""
        self.task_performance[task_id].append({
            'timestamp': time.time(),
            'accuracy': accuracy
        })
        
        # Keep only recent records
        if len(self.task_performance[task_id]) > 100:
            self.task_performance[task_id] = self.task_performance[task_id][-100:]
    
    def _get_recent_accuracy(self) -> Optional[float]:
        """Get recent average accuracy across all tasks."""
        recent_accuracies = []
        cutoff_time = time.time() - 3600  # Last hour
        
        for task_id, records in self.task_performance.items():
            recent_records = [r for r in records if r['timestamp'] > cutoff_time]
            if recent_records:
                recent_accuracies.extend([r['accuracy'] for r in recent_records])
        
        return np.mean(recent_accuracies) if recent_accuracies else None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics collected yet"}
        
        latest_metrics = self.metrics_history[-1]
        health_status = self.check_system_health(latest_metrics)
        
        return {
            "health": health_status.overall_health,
            "components": health_status.components,
            "alerts": health_status.alerts,
            "recommendations": health_status.recommendations,
            "metrics": latest_metrics.to_dict(),
            "uptime": time.time() - (self.metrics_history[0].timestamp if self.metrics_history else time.time())
        }
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export collected metrics to file."""
        filepath = Path(filepath)
        
        metrics_data = [metrics.to_dict() for metrics in self.metrics_history]
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        elif format.lower() == "csv":
            import pandas as pd
            df = pd.DataFrame(metrics_data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Metrics exported to {filepath}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        # Compute statistics
        metrics_data = [m.to_dict() for m in self.metrics_history]
        
        report = {
            "summary": {
                "total_samples": len(metrics_data),
                "time_range": {
                    "start": min(m["timestamp"] for m in metrics_data),
                    "end": max(m["timestamp"] for m in metrics_data)
                }
            },
            "performance": {},
            "resource_usage": {},
            "health_trends": {}
        }
        
        # Performance statistics
        inference_times = [m["inference_time_ms"] for m in metrics_data if m["inference_time_ms"] > 0]
        if inference_times:
            report["performance"]["inference_time"] = {
                "mean": np.mean(inference_times),
                "std": np.std(inference_times),
                "min": np.min(inference_times),
                "max": np.max(inference_times),
                "p95": np.percentile(inference_times, 95)
            }
        
        throughputs = [m["throughput"] for m in metrics_data if m["throughput"] > 0]
        if throughputs:
            report["performance"]["throughput"] = {
                "mean": np.mean(throughputs),
                "std": np.std(throughputs),
                "min": np.min(throughputs),
                "max": np.max(throughputs)
            }
        
        # Resource usage statistics
        cpu_usage = [m["cpu_percent"] for m in metrics_data]
        memory_usage = [m["memory_percent"] for m in metrics_data]
        
        report["resource_usage"]["cpu"] = {
            "mean": np.mean(cpu_usage),
            "max": np.max(cpu_usage),
            "min": np.min(cpu_usage)
        }
        
        report["resource_usage"]["memory"] = {
            "mean": np.mean(memory_usage),
            "max": np.max(memory_usage),
            "min": np.min(memory_usage)
        }
        
        # GPU statistics if available
        gpu_memory_data = [m["gpu_memory_used_mb"] for m in metrics_data if m["gpu_memory_used_mb"] is not None]
        if gpu_memory_data:
            report["resource_usage"]["gpu_memory"] = {
                "mean": np.mean(gpu_memory_data),
                "max": np.max(gpu_memory_data),
                "min": np.min(gpu_memory_data)
            }
        
        return report


class PerformanceProfiler:
    """Detailed performance profiling for model components."""
    
    def __init__(self, model):
        self.model = model
        self.profiles = {}
        self.current_profile = None
        
    def start_profiling(self, profile_name: str):
        """Start a new profiling session."""
        self.current_profile = {
            "name": profile_name,
            "start_time": time.time(),
            "events": [],
            "memory_snapshots": []
        }
        
        # Initial memory snapshot
        self._take_memory_snapshot("start")
    
    def record_event(self, event_name: str, duration_ms: float, metadata: Dict = None):
        """Record a profiling event."""
        if self.current_profile is None:
            return
        
        event = {
            "name": event_name,
            "timestamp": time.time(),
            "duration_ms": duration_ms,
            "metadata": metadata or {}
        }
        self.current_profile["events"].append(event)
    
    def _take_memory_snapshot(self, label: str):
        """Take a memory usage snapshot."""
        if self.current_profile is None:
            return
        
        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "cpu_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        if torch.cuda.is_available():
            snapshot["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        
        self.current_profile["memory_snapshots"].append(snapshot)
    
    def end_profiling(self) -> Dict[str, Any]:
        """End current profiling session and return results."""
        if self.current_profile is None:
            return {}
        
        self._take_memory_snapshot("end")
        
        profile = self.current_profile
        profile["end_time"] = time.time()
        profile["total_duration"] = profile["end_time"] - profile["start_time"]
        
        # Store profile
        self.profiles[profile["name"]] = profile
        self.current_profile = None
        
        return profile
    
    def get_profile_summary(self, profile_name: str) -> Dict[str, Any]:
        """Get summary of a completed profile."""
        if profile_name not in self.profiles:
            return {}
        
        profile = self.profiles[profile_name]
        events = profile["events"]
        
        if not events:
            return {"error": "No events recorded"}
        
        # Event statistics
        event_durations = [e["duration_ms"] for e in events]
        
        summary = {
            "total_events": len(events),
            "total_duration": sum(event_durations),
            "mean_event_duration": np.mean(event_durations),
            "max_event_duration": max(event_durations),
            "event_breakdown": {}
        }
        
        # Breakdown by event type
        event_types = defaultdict(list)
        for event in events:
            event_types[event["name"]].append(event["duration_ms"])
        
        for event_type, durations in event_types.items():
            summary["event_breakdown"][event_type] = {
                "count": len(durations),
                "total_duration": sum(durations),
                "mean_duration": np.mean(durations),
                "percentage": (sum(durations) / summary["total_duration"]) * 100
            }
        
        # Memory analysis
        snapshots = profile["memory_snapshots"]
        if len(snapshots) >= 2:
            start_mem = snapshots[0]["cpu_memory_mb"]
            end_mem = snapshots[-1]["cpu_memory_mb"]
            summary["memory_delta_mb"] = end_mem - start_mem
            
            if "gpu_memory_mb" in snapshots[0]:
                start_gpu = snapshots[0]["gpu_memory_mb"]
                end_gpu = snapshots[-1]["gpu_memory_mb"]
                summary["gpu_memory_delta_mb"] = end_gpu - start_gpu
        
        return summary