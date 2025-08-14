"""
Auto-scaling system for continual learning workloads.
Provides dynamic resource allocation, load balancing, and performance optimization.
"""

import time
import threading
import logging
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from pathlib import Path
import json
from datetime import datetime, timedelta
from enum import Enum
import subprocess
import yaml

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    OPTIMIZE = "optimize"
    MIGRATE = "migrate"
    NO_ACTION = "no_action"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    gpu_memory_usage: Optional[float]
    network_io: float
    disk_io: float
    active_tasks: int
    queue_length: int
    inference_latency: float
    throughput: float


@dataclass
class ScalingRule:
    """Definition of a scaling rule."""
    name: str
    resource_type: ResourceType
    metric_threshold: float
    duration_seconds: float
    action: ScalingAction
    priority: int
    cooldown_seconds: float
    conditions: Dict[str, Any]


@dataclass
class WorkloadPrediction:
    """Predicted workload characteristics."""
    timestamp: float
    predicted_load: float
    confidence: float
    time_horizon_minutes: int
    resource_requirements: Dict[str, float]


class WorkloadPredictor:
    """Predicts future workload based on historical patterns."""
    
    def __init__(self, history_window_hours: int = 24):
        self.history_window_hours = history_window_hours
        self.metrics_history = deque(maxlen=history_window_hours * 60)  # Minute-level data
        self.patterns = {}
        
    def add_metrics(self, metrics: ResourceMetrics):
        """Add new metrics to history."""
        self.metrics_history.append(metrics)
        self._update_patterns()
    
    def _update_patterns(self):
        """Update learned patterns from historical data."""
        if len(self.metrics_history) < 60:  # Need at least 1 hour of data
            return
        
        # Extract features
        recent_metrics = list(self.metrics_history)[-60:]  # Last hour
        
        # Calculate patterns
        self.patterns = {
            "avg_cpu": np.mean([m.cpu_usage for m in recent_metrics]),
            "avg_memory": np.mean([m.memory_usage for m in recent_metrics]),
            "avg_throughput": np.mean([m.throughput for m in recent_metrics]),
            "peak_hours": self._identify_peak_hours(),
            "load_trend": self._calculate_load_trend(),
            "seasonality": self._detect_seasonality()
        }
    
    def _identify_peak_hours(self) -> List[int]:
        """Identify peak usage hours."""
        if len(self.metrics_history) < 24 * 60:  # Need at least 24 hours
            return []
        
        hourly_loads = defaultdict(list)
        
        for metrics in self.metrics_history:
            hour = datetime.fromtimestamp(metrics.timestamp).hour
            load = (metrics.cpu_usage + metrics.memory_usage) / 2
            hourly_loads[hour].append(load)
        
        # Calculate average load per hour
        avg_hourly_loads = {
            hour: np.mean(loads) for hour, loads in hourly_loads.items()
        }
        
        # Identify hours with above-average load
        overall_avg = np.mean(list(avg_hourly_loads.values()))
        peak_hours = [
            hour for hour, load in avg_hourly_loads.items()
            if load > overall_avg * 1.2
        ]
        
        return peak_hours
    
    def _calculate_load_trend(self) -> float:
        """Calculate load trend (positive = increasing, negative = decreasing)."""
        if len(self.metrics_history) < 30:
            return 0.0
        
        recent_loads = [
            (m.cpu_usage + m.memory_usage) / 2
            for m in list(self.metrics_history)[-30:]
        ]
        
        # Simple linear trend
        x = np.arange(len(recent_loads))
        trend = np.polyfit(x, recent_loads, 1)[0]
        
        return trend
    
    def _detect_seasonality(self) -> Dict[str, float]:
        """Detect seasonal patterns in workload."""
        if len(self.metrics_history) < 24 * 60 * 7:  # Need at least a week
            return {}
        
        # Daily seasonality
        daily_pattern = defaultdict(list)
        
        for metrics in self.metrics_history:
            dt = datetime.fromtimestamp(metrics.timestamp)
            hour = dt.hour
            load = (metrics.cpu_usage + metrics.memory_usage) / 2
            daily_pattern[hour].append(load)
        
        daily_seasonality = {
            f"hour_{hour}": np.mean(loads)
            for hour, loads in daily_pattern.items()
        }
        
        return {"daily": daily_seasonality}
    
    def predict_workload(self, time_horizon_minutes: int = 60) -> WorkloadPrediction:
        """Predict workload for the specified time horizon."""
        
        if not self.patterns:
            # No patterns learned yet, return current state
            if self.metrics_history:
                current = self.metrics_history[-1]
                return WorkloadPrediction(
                    timestamp=time.time(),
                    predicted_load=(current.cpu_usage + current.memory_usage) / 2,
                    confidence=0.5,
                    time_horizon_minutes=time_horizon_minutes,
                    resource_requirements={
                        "cpu": current.cpu_usage,
                        "memory": current.memory_usage,
                        "gpu": current.gpu_usage or 0
                    }
                )
            else:
                return WorkloadPrediction(
                    timestamp=time.time(),
                    predicted_load=50.0,
                    confidence=0.1,
                    time_horizon_minutes=time_horizon_minutes,
                    resource_requirements={"cpu": 50, "memory": 50, "gpu": 0}
                )
        
        # Predict based on learned patterns
        current_time = datetime.now()
        future_time = current_time + timedelta(minutes=time_horizon_minutes)
        
        # Base prediction on current trend
        base_load = self.patterns.get("avg_cpu", 50.0)
        trend_adjustment = self.patterns.get("load_trend", 0) * time_horizon_minutes
        
        # Adjust for peak hours
        peak_hours = self.patterns.get("peak_hours", [])
        if future_time.hour in peak_hours:
            peak_adjustment = 20.0  # 20% increase during peak hours
        else:
            peak_adjustment = 0.0
        
        # Seasonal adjustment
        seasonality = self.patterns.get("seasonality", {})
        daily_pattern = seasonality.get("daily", {})
        seasonal_adjustment = daily_pattern.get(f"hour_{future_time.hour}", base_load) - base_load
        
        # Final prediction
        predicted_load = max(0, min(100, base_load + trend_adjustment + peak_adjustment + seasonal_adjustment))
        
        # Confidence based on data availability and pattern consistency
        confidence = min(1.0, len(self.metrics_history) / (24 * 60))  # Higher with more data
        
        # Resource requirements based on predicted load
        resource_requirements = {
            "cpu": predicted_load,
            "memory": predicted_load * 0.8,  # Memory typically lower than CPU
            "gpu": predicted_load * 0.6 if self.patterns.get("avg_gpu", 0) > 0 else 0
        }
        
        return WorkloadPrediction(
            timestamp=time.time(),
            predicted_load=predicted_load,
            confidence=confidence,
            time_horizon_minutes=time_horizon_minutes,
            resource_requirements=resource_requirements
        )


class AutoScaler:
    """Automatic scaling system for continual learning workloads."""
    
    def __init__(
        self,
        model,
        config,
        scaling_rules: Optional[List[ScalingRule]] = None,
        monitoring_interval: float = 30.0
    ):
        self.model = model
        self.config = config
        self.monitoring_interval = monitoring_interval
        
        # Scaling rules
        self.scaling_rules = scaling_rules or self._create_default_rules()
        
        # Workload prediction
        self.workload_predictor = WorkloadPredictor()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)
        
        # Scaling state
        self.last_scaling_action = {}
        self.scaling_cooldowns = defaultdict(float)
        self.active_scaling_operations = {}
        
        # Performance tracking
        self.scaling_decisions = []
        self.cost_optimization_metrics = defaultdict(list)
        
        logger.info("AutoScaler initialized")
    
    def _create_default_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        
        return [
            # CPU scaling rules
            ScalingRule(
                name="cpu_scale_up",
                resource_type=ResourceType.CPU,
                metric_threshold=80.0,
                duration_seconds=300,  # 5 minutes
                action=ScalingAction.SCALE_UP,
                priority=1,
                cooldown_seconds=600,  # 10 minutes
                conditions={"queue_length": 5}
            ),
            ScalingRule(
                name="cpu_scale_down",
                resource_type=ResourceType.CPU,
                metric_threshold=20.0,
                duration_seconds=900,  # 15 minutes
                action=ScalingAction.SCALE_DOWN,
                priority=2,
                cooldown_seconds=1800,  # 30 minutes
                conditions={"queue_length": 0}
            ),
            
            # Memory scaling rules
            ScalingRule(
                name="memory_scale_up",
                resource_type=ResourceType.MEMORY,
                metric_threshold=85.0,
                duration_seconds=180,  # 3 minutes
                action=ScalingAction.SCALE_UP,
                priority=1,
                cooldown_seconds=300,  # 5 minutes
                conditions={}
            ),
            
            # GPU scaling rules
            ScalingRule(
                name="gpu_scale_up",
                resource_type=ResourceType.GPU,
                metric_threshold=90.0,
                duration_seconds=240,  # 4 minutes
                action=ScalingAction.SCALE_UP,
                priority=1,
                cooldown_seconds=600,  # 10 minutes
                conditions={"gpu_available": True}
            ),
            
            # Performance optimization rules
            ScalingRule(
                name="latency_optimize",
                resource_type=ResourceType.CPU,
                metric_threshold=1000.0,  # 1 second latency
                duration_seconds=120,  # 2 minutes
                action=ScalingAction.OPTIMIZE,
                priority=3,
                cooldown_seconds=300,  # 5 minutes
                conditions={}
            )
        ]
    
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        
        if self.is_monitoring:
            logger.warning("Auto-scaling already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        
        while self.is_monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_resource_metrics()
                self.metrics_history.append(metrics)
                self.workload_predictor.add_metrics(metrics)
                
                # Evaluate scaling rules
                scaling_decision = self._evaluate_scaling_rules(metrics)
                
                if scaling_decision["action"] != ScalingAction.NO_ACTION:
                    # Execute scaling action
                    success = self._execute_scaling_action(scaling_decision)
                    
                    # Record decision
                    self.scaling_decisions.append({
                        "timestamp": time.time(),
                        "decision": scaling_decision,
                        "success": success,
                        "metrics": asdict(metrics)
                    })
                
                # Predictive scaling
                self._evaluate_predictive_scaling()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource utilization metrics."""
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Network and disk I/O
        net_io = psutil.net_io_counters()
        disk_io = psutil.disk_io_counters()
        
        network_io = net_io.bytes_sent + net_io.bytes_recv if net_io else 0
        disk_io_bytes = disk_io.read_bytes + disk_io.write_bytes if disk_io else 0
        
        # GPU metrics
        gpu_usage = None
        gpu_memory_usage = None
        
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                pass
        
        # Model-specific metrics
        active_tasks = 0
        queue_length = 0
        inference_latency = 0.0
        throughput = 0.0
        
        if hasattr(self.model, 'get_system_status'):
            try:
                status = self.model.get_system_status()
                # Extract relevant metrics from status
                active_tasks = len(status.get("model_info", {}).get("adapters", []))
                
                if hasattr(self.model, 'system_monitor'):
                    monitor = self.model.system_monitor
                    if hasattr(monitor, 'metrics_history') and monitor.metrics_history:
                        latest = monitor.metrics_history[-1]
                        inference_latency = getattr(latest, 'inference_latency_ms', 0.0)
                
            except Exception as e:
                logger.debug(f"Could not get model metrics: {e}")
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            network_io=float(network_io),
            disk_io=float(disk_io_bytes),
            active_tasks=active_tasks,
            queue_length=queue_length,
            inference_latency=inference_latency,
            throughput=throughput
        )
    
    def _evaluate_scaling_rules(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Evaluate scaling rules against current metrics."""
        
        triggered_rules = []
        
        for rule in self.scaling_rules:
            # Check cooldown
            if time.time() - self.scaling_cooldowns[rule.name] < rule.cooldown_seconds:
                continue
            
            # Check conditions
            if not self._check_rule_conditions(rule, metrics):
                continue
            
            # Check threshold
            if self._check_rule_threshold(rule, metrics):
                triggered_rules.append(rule)
        
        if not triggered_rules:
            return {"action": ScalingAction.NO_ACTION}
        
        # Select highest priority rule
        selected_rule = min(triggered_rules, key=lambda r: r.priority)
        
        # Check if rule has been triggered for sufficient duration
        if not self._check_rule_duration(selected_rule, metrics):
            return {"action": ScalingAction.NO_ACTION}
        
        return {
            "action": selected_rule.action,
            "rule": selected_rule,
            "resource_type": selected_rule.resource_type,
            "current_value": self._get_metric_value(selected_rule.resource_type, metrics),
            "threshold": selected_rule.metric_threshold
        }
    
    def _check_rule_conditions(self, rule: ScalingRule, metrics: ResourceMetrics) -> bool:
        """Check if rule conditions are met."""
        
        for condition, expected_value in rule.conditions.items():
            if condition == "queue_length":
                if metrics.queue_length < expected_value:
                    return False
            elif condition == "gpu_available":
                if expected_value and not torch.cuda.is_available():
                    return False
            elif condition == "active_tasks":
                if metrics.active_tasks < expected_value:
                    return False
        
        return True
    
    def _check_rule_threshold(self, rule: ScalingRule, metrics: ResourceMetrics) -> bool:
        """Check if metric exceeds rule threshold."""
        
        current_value = self._get_metric_value(rule.resource_type, metrics)
        
        if rule.action in [ScalingAction.SCALE_UP, ScalingAction.OPTIMIZE]:
            return current_value > rule.metric_threshold
        elif rule.action == ScalingAction.SCALE_DOWN:
            return current_value < rule.metric_threshold
        
        return False
    
    def _check_rule_duration(self, rule: ScalingRule, metrics: ResourceMetrics) -> bool:
        """Check if threshold has been exceeded for sufficient duration."""
        
        if len(self.metrics_history) < 2:
            return False
        
        # Check how long the condition has been true
        duration_start = None
        current_time = metrics.timestamp
        
        for i in range(len(self.metrics_history) - 1, -1, -1):
            historical_metrics = self.metrics_history[i]
            
            if self._check_rule_threshold(rule, historical_metrics):
                duration_start = historical_metrics.timestamp
            else:
                break
        
        if duration_start is None:
            return False
        
        duration = current_time - duration_start
        return duration >= rule.duration_seconds
    
    def _get_metric_value(self, resource_type: ResourceType, metrics: ResourceMetrics) -> float:
        """Get metric value for specific resource type."""
        
        if resource_type == ResourceType.CPU:
            return metrics.cpu_usage
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_usage
        elif resource_type == ResourceType.GPU:
            return metrics.gpu_usage or 0.0
        elif resource_type == ResourceType.NETWORK:
            return metrics.network_io
        elif resource_type == ResourceType.STORAGE:
            return metrics.disk_io
        
        return 0.0
    
    def _execute_scaling_action(self, decision: Dict[str, Any]) -> bool:
        """Execute the determined scaling action."""
        
        action = decision["action"]
        rule = decision["rule"]
        
        logger.info(f"Executing scaling action: {action.value} for {rule.resource_type.value}")
        
        try:
            if action == ScalingAction.SCALE_UP:
                return self._scale_up(rule, decision)
            elif action == ScalingAction.SCALE_DOWN:
                return self._scale_down(rule, decision)
            elif action == ScalingAction.OPTIMIZE:
                return self._optimize_performance(rule, decision)
            elif action == ScalingAction.MIGRATE:
                return self._migrate_workload(rule, decision)
            
            return False
            
        except Exception as e:
            logger.error(f"Scaling action failed: {e}")
            return False
        
        finally:
            # Update cooldown
            self.scaling_cooldowns[rule.name] = time.time()
    
    def _scale_up(self, rule: ScalingRule, decision: Dict[str, Any]) -> bool:
        """Scale up resources."""
        
        resource_type = rule.resource_type
        
        if resource_type == ResourceType.CPU:
            # Increase CPU allocation (in container/cloud environment)
            return self._scale_cpu_up()
        elif resource_type == ResourceType.MEMORY:
            # Increase memory allocation
            return self._scale_memory_up()
        elif resource_type == ResourceType.GPU:
            # Add GPU resources
            return self._scale_gpu_up()
        
        return False
    
    def _scale_down(self, rule: ScalingRule, decision: Dict[str, Any]) -> bool:
        """Scale down resources."""
        
        resource_type = rule.resource_type
        
        if resource_type == ResourceType.CPU:
            return self._scale_cpu_down()
        elif resource_type == ResourceType.MEMORY:
            return self._scale_memory_down()
        elif resource_type == ResourceType.GPU:
            return self._scale_gpu_down()
        
        return False
    
    def _optimize_performance(self, rule: ScalingRule, decision: Dict[str, Any]) -> bool:
        """Optimize model performance."""
        
        try:
            if hasattr(self.model, 'optimize_for_inference'):
                optimizations = self.model.optimize_for_inference("speed")
                logger.info(f"Applied performance optimizations: {optimizations}")
                return True
            
            # Other optimization strategies
            self._optimize_batch_size()
            self._optimize_model_settings()
            
            return True
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            return False
    
    def _migrate_workload(self, rule: ScalingRule, decision: Dict[str, Any]) -> bool:
        """Migrate workload to different resources."""
        
        # Placeholder for workload migration logic
        logger.info("Workload migration not implemented")
        return False
    
    def _scale_cpu_up(self) -> bool:
        """Scale up CPU resources."""
        
        # In a cloud environment, this would call APIs to increase CPU allocation
        # In Kubernetes, this would update resource requests/limits
        # For now, we optimize CPU usage
        
        try:
            # Enable CPU optimizations
            if hasattr(self.config, 'num_workers'):
                self.config.num_workers = min(psutil.cpu_count(), self.config.num_workers + 1)
            
            # Set CPU affinity if possible
            try:
                import os
                os.sched_setaffinity(0, list(range(psutil.cpu_count())))
            except:
                pass
            
            logger.info("CPU scaling up completed")
            return True
            
        except Exception as e:
            logger.error(f"CPU scale up failed: {e}")
            return False
    
    def _scale_cpu_down(self) -> bool:
        """Scale down CPU resources."""
        
        try:
            if hasattr(self.config, 'num_workers'):
                self.config.num_workers = max(1, self.config.num_workers - 1)
            
            logger.info("CPU scaling down completed")
            return True
            
        except Exception as e:
            logger.error(f"CPU scale down failed: {e}")
            return False
    
    def _scale_memory_up(self) -> bool:
        """Scale up memory resources."""
        
        try:
            # Adjust memory-related settings
            if hasattr(self.config, 'max_sequence_length'):
                # Don't increase further if already high
                if self.config.max_sequence_length < 512:
                    self.config.max_sequence_length = min(512, self.config.max_sequence_length + 64)
            
            # Enable memory optimizations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Memory scaling up completed")
            return True
            
        except Exception as e:
            logger.error(f"Memory scale up failed: {e}")
            return False
    
    def _scale_memory_down(self) -> bool:
        """Scale down memory usage."""
        
        try:
            # Reduce memory usage
            if hasattr(self.config, 'batch_size'):
                self.config.batch_size = max(1, self.config.batch_size // 2)
            
            if hasattr(self.config, 'max_sequence_length'):
                self.config.max_sequence_length = max(64, self.config.max_sequence_length - 64)
            
            # Clear caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()
            
            logger.info("Memory scaling down completed")
            return True
            
        except Exception as e:
            logger.error(f"Memory scale down failed: {e}")
            return False
    
    def _scale_gpu_up(self) -> bool:
        """Scale up GPU resources."""
        
        if not torch.cuda.is_available():
            return False
        
        try:
            # Enable GPU optimizations
            if hasattr(self.config, 'mixed_precision'):
                self.config.mixed_precision = True
            
            # Move model to GPU if not already
            if self.model.device.type == "cpu":
                self.model.to(torch.device("cuda"))
                self.config.device = torch.device("cuda")
            
            logger.info("GPU scaling up completed")
            return True
            
        except Exception as e:
            logger.error(f"GPU scale up failed: {e}")
            return False
    
    def _scale_gpu_down(self) -> bool:
        """Scale down GPU usage."""
        
        try:
            # Move to CPU if needed
            if self.config.device.type == "cuda":
                self.model.cpu()
                self.config.device = torch.device("cpu")
            
            # Disable GPU-specific optimizations
            if hasattr(self.config, 'mixed_precision'):
                self.config.mixed_precision = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("GPU scaling down completed")
            return True
            
        except Exception as e:
            logger.error(f"GPU scale down failed: {e}")
            return False
    
    def _optimize_batch_size(self):
        """Dynamically optimize batch size."""
        
        if not hasattr(self.config, 'batch_size'):
            return
        
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = list(self.metrics_history)[-10:]
        avg_memory_usage = np.mean([m.memory_usage for m in recent_metrics])
        avg_latency = np.mean([m.inference_latency for m in recent_metrics])
        
        # Adjust batch size based on memory usage and latency
        if avg_memory_usage < 60 and avg_latency < 500:
            # Can increase batch size
            self.config.batch_size = min(64, self.config.batch_size * 2)
        elif avg_memory_usage > 80 or avg_latency > 1000:
            # Should decrease batch size
            self.config.batch_size = max(1, self.config.batch_size // 2)
    
    def _optimize_model_settings(self):
        """Optimize model-specific settings."""
        
        # Enable gradient checkpointing if memory is high
        if len(self.metrics_history) > 0:
            latest_metrics = self.metrics_history[-1]
            
            if latest_metrics.memory_usage > 80:
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
    
    def _evaluate_predictive_scaling(self):
        """Evaluate need for predictive scaling based on workload prediction."""
        
        prediction = self.workload_predictor.predict_workload(time_horizon_minutes=30)
        
        if prediction.confidence < 0.6:
            return  # Not confident enough in prediction
        
        # Check if predicted load significantly differs from current
        if len(self.metrics_history) > 0:
            current_metrics = self.metrics_history[-1]
            current_load = (current_metrics.cpu_usage + current_metrics.memory_usage) / 2
            
            load_increase = prediction.predicted_load - current_load
            
            # Preemptive scaling if significant load increase predicted
            if load_increase > 30:
                logger.info(f"Preemptive scaling triggered: predicted load increase of {load_increase:.1f}%")
                
                # Prepare for increased load
                self._preemptive_scale_up(prediction)
            elif load_increase < -30:
                logger.info(f"Preemptive scale down: predicted load decrease of {abs(load_increase):.1f}%")
                
                # Prepare for decreased load
                self._preemptive_scale_down(prediction)
    
    def _preemptive_scale_up(self, prediction: WorkloadPrediction):
        """Preemptively scale up based on prediction."""
        
        # Increase resources based on predicted requirements
        resource_reqs = prediction.resource_requirements
        
        if resource_reqs.get("cpu", 0) > 70:
            self._scale_cpu_up()
        
        if resource_reqs.get("memory", 0) > 75:
            self._scale_memory_up()
        
        if resource_reqs.get("gpu", 0) > 70 and torch.cuda.is_available():
            self._scale_gpu_up()
    
    def _preemptive_scale_down(self, prediction: WorkloadPrediction):
        """Preemptively scale down based on prediction."""
        
        # Reduce resources based on predicted requirements
        resource_reqs = prediction.resource_requirements
        
        if resource_reqs.get("cpu", 100) < 30:
            self._scale_cpu_down()
        
        if resource_reqs.get("memory", 100) < 40:
            self._scale_memory_down()
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get comprehensive scaling summary."""
        
        recent_decisions = [
            d for d in self.scaling_decisions
            if d["timestamp"] > time.time() - 3600  # Last hour
        ]
        
        # Calculate scaling efficiency
        successful_scalings = sum(1 for d in recent_decisions if d["success"])
        scaling_success_rate = successful_scalings / max(len(recent_decisions), 1)
        
        # Resource utilization trends
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-60:]  # Last hour
            
            utilization_trends = {
                "cpu": {
                    "current": recent_metrics[-1].cpu_usage,
                    "avg": np.mean([m.cpu_usage for m in recent_metrics]),
                    "trend": "increasing" if len(recent_metrics) > 1 and 
                            recent_metrics[-1].cpu_usage > recent_metrics[0].cpu_usage else "stable"
                },
                "memory": {
                    "current": recent_metrics[-1].memory_usage,
                    "avg": np.mean([m.memory_usage for m in recent_metrics]),
                    "trend": "increasing" if len(recent_metrics) > 1 and 
                            recent_metrics[-1].memory_usage > recent_metrics[0].memory_usage else "stable"
                }
            }
        else:
            utilization_trends = {}
        
        # Workload prediction
        prediction = self.workload_predictor.predict_workload()
        
        return {
            "scaling_decisions_last_hour": len(recent_decisions),
            "scaling_success_rate": scaling_success_rate,
            "utilization_trends": utilization_trends,
            "workload_prediction": asdict(prediction),
            "active_rules": len(self.scaling_rules),
            "cost_optimization": {
                "total_optimizations": len(self.cost_optimization_metrics),
                "estimated_savings": sum(self.cost_optimization_metrics.get("savings", []))
            }
        }
    
    def cleanup(self):
        """Clean up auto-scaling resources."""
        
        self.stop_monitoring()
        self.metrics_history.clear()
        self.scaling_decisions.clear()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock model and config for testing
    class MockModel:
        def __init__(self):
            self.device = torch.device("cpu")
        
        def get_system_status(self):
            return {"model_info": {"adapters": ["task1", "task2"]}}
        
        def optimize_for_inference(self, level):
            return {"optimization_applied": level}
        
        def cpu(self):
            self.device = torch.device("cpu")
        
        def to(self, device):
            self.device = device
    
    class MockConfig:
        def __init__(self):
            self.batch_size = 16
            self.num_workers = 2
            self.device = torch.device("cpu")
            self.max_sequence_length = 128
    
    # Test auto-scaler
    model = MockModel()
    config = MockConfig()
    
    scaler = AutoScaler(model, config)
    scaler.start_monitoring()
    
    # Let it run for a few seconds
    time.sleep(10)
    
    # Get summary
    summary = scaler.get_scaling_summary()
    print(f"Scaling summary: {summary}")
    
    # Cleanup
    scaler.cleanup()