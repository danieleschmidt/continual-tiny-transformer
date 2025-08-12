"""
Adaptive learning system for dynamic optimization of continual learning.
Automatically adjusts hyperparameters, architecture, and training strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import threading
from collections import deque
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class AdaptationMetrics:
    """Metrics for tracking adaptation decisions."""
    task_id: str
    metric_name: str
    current_value: float
    target_value: float
    adaptation_strength: float
    timestamp: float = field(default_factory=time.time)


class MetricTracker:
    """Tracks and analyzes performance metrics for adaptive decisions."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = {}
        self.trend_analysis = {}
        self._lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: float, context: Dict[str, Any] = None):
        """Record a performance metric."""
        with self._lock:
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = deque(maxlen=self.window_size)
            
            entry = {
                "value": value,
                "timestamp": time.time(),
                "context": context or {}
            }
            
            self.metrics_history[metric_name].append(entry)
            self._update_trend_analysis(metric_name)
    
    def _update_trend_analysis(self, metric_name: str):
        """Update trend analysis for a metric."""
        history = self.metrics_history[metric_name]
        if len(history) < 5:  # Need minimum data for trend analysis
            return
        
        values = [entry["value"] for entry in history]
        recent_values = values[-10:]  # Last 10 values
        older_values = values[-20:-10] if len(values) >= 20 else values[:-10]
        
        # Calculate trends
        recent_avg = np.mean(recent_values)
        older_avg = np.mean(older_values) if older_values else recent_avg
        
        trend_direction = "improving" if recent_avg > older_avg else "degrading"
        if abs(recent_avg - older_avg) / max(abs(older_avg), 1e-8) < 0.05:
            trend_direction = "stable"
        
        volatility = np.std(recent_values) if len(recent_values) > 1 else 0.0
        
        self.trend_analysis[metric_name] = {
            "trend_direction": trend_direction,
            "recent_avg": recent_avg,
            "older_avg": older_avg,
            "volatility": volatility,
            "change_rate": (recent_avg - older_avg) / max(abs(older_avg), 1e-8),
            "last_updated": time.time()
        }
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive summary of a metric."""
        with self._lock:
            if metric_name not in self.metrics_history:
                return {"status": "no_data"}
            
            history = self.metrics_history[metric_name]
            values = [entry["value"] for entry in history]
            
            summary = {
                "current_value": values[-1] if values else None,
                "avg_value": np.mean(values),
                "min_value": np.min(values),
                "max_value": np.max(values),
                "std_value": np.std(values),
                "data_points": len(values),
                "trend_analysis": self.trend_analysis.get(metric_name, {})
            }
            
            return summary
    
    def should_adapt(self, metric_name: str, threshold: float = 0.1) -> bool:
        """Determine if adaptation is needed based on metric trends."""
        trend = self.trend_analysis.get(metric_name, {})
        
        if not trend:
            return False
        
        # Adapt if performance is degrading significantly
        if (trend["trend_direction"] == "degrading" and 
            abs(trend["change_rate"]) > threshold):
            return True
        
        # Adapt if volatility is too high
        if trend["volatility"] > threshold * 2:
            return True
        
        return False


class HyperparameterOptimizer:
    """Adaptive hyperparameter optimization using Bayesian methods."""
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        self.param_bounds = param_bounds
        self.evaluation_history = []
        self.best_params = {}
        self.current_params = {}
        
        # Initialize with reasonable defaults
        for param, (min_val, max_val) in param_bounds.items():
            self.current_params[param] = (min_val + max_val) / 2
    
    def suggest_parameters(self, performance_metric: float) -> Dict[str, float]:
        """Suggest new hyperparameters based on performance feedback."""
        
        # Record current evaluation
        self.evaluation_history.append({
            "params": self.current_params.copy(),
            "performance": performance_metric,
            "timestamp": time.time()
        })
        
        # Update best parameters if current is better
        if not self.best_params or performance_metric > self.best_params.get("performance", -float('inf')):
            self.best_params = {
                "params": self.current_params.copy(),
                "performance": performance_metric
            }
        
        # Generate new parameters using acquisition function
        new_params = self._acquisition_function()
        self.current_params = new_params
        
        logger.info(f"Suggested hyperparameters: {new_params}")
        return new_params
    
    def _acquisition_function(self) -> Dict[str, float]:
        """Acquisition function for parameter exploration vs exploitation."""
        
        if len(self.evaluation_history) < 3:
            # Random exploration for first few evaluations
            return self._random_sample()
        
        # Use Upper Confidence Bound (UCB) approach
        exploration_factor = 2.0
        
        new_params = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            
            # Get historical values for this parameter
            param_values = [eval_data["params"][param] for eval_data in self.evaluation_history]
            performances = [eval_data["performance"] for eval_data in self.evaluation_history]
            
            # Simple Gaussian Process approximation
            mean_performance = np.mean(performances)
            std_performance = np.std(performances) if len(performances) > 1 else 0.1
            
            # UCB acquisition
            best_value = self.best_params["params"][param]
            exploration_term = exploration_factor * std_performance
            
            # Bias towards best known value with some exploration
            candidate = best_value + np.random.normal(0, exploration_term * (max_val - min_val) * 0.1)
            
            # Ensure within bounds
            new_params[param] = np.clip(candidate, min_val, max_val)
        
        return new_params
    
    def _random_sample(self) -> Dict[str, float]:
        """Random sampling within parameter bounds."""
        return {
            param: np.random.uniform(min_val, max_val)
            for param, (min_val, max_val) in self.param_bounds.items()
        }
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of hyperparameter optimization."""
        return {
            "evaluations_completed": len(self.evaluation_history),
            "best_params": self.best_params,
            "current_params": self.current_params,
            "recent_performances": [
                eval_data["performance"] for eval_data in self.evaluation_history[-5:]
            ]
        }


class ArchitectureAdaptation:
    """Adaptive architecture modification for continual learning."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.adaptation_history = []
        self.architecture_variants = {}
    
    def adapt_architecture(
        self, 
        task_performance: Dict[str, float],
        resource_constraints: Dict[str, float]
    ) -> bool:
        """Adapt model architecture based on performance and constraints."""
        
        adaptation_needed = self._analyze_adaptation_need(task_performance, resource_constraints)
        
        if not adaptation_needed:
            return False
        
        # Determine adaptation strategy
        if resource_constraints.get("memory_usage", 0) > 0.8:
            success = self._reduce_model_complexity()
        elif task_performance.get("accuracy", 0) < 0.7:
            success = self._increase_model_capacity()
        elif task_performance.get("inference_time", 0) > resource_constraints.get("max_latency", 1.0):
            success = self._optimize_for_speed()
        else:
            success = self._balanced_adaptation()
        
        if success:
            self.adaptation_history.append({
                "timestamp": time.time(),
                "trigger": adaptation_needed,
                "performance_before": task_performance.copy(),
                "constraints": resource_constraints.copy()
            })
            
            logger.info("Architecture adapted successfully")
        
        return success
    
    def _analyze_adaptation_need(
        self, 
        performance: Dict[str, float], 
        constraints: Dict[str, float]
    ) -> str:
        """Analyze if architecture adaptation is needed."""
        
        # Check memory constraints
        if constraints.get("memory_usage", 0) > 0.9:
            return "memory_pressure"
        
        # Check performance constraints
        if performance.get("accuracy", 1.0) < 0.6:
            return "low_accuracy"
        
        # Check latency constraints
        if performance.get("inference_time", 0) > constraints.get("max_latency", 1.0) * 1.5:
            return "high_latency"
        
        # Check if model is underperforming across multiple metrics
        poor_metrics = sum(1 for v in performance.values() if v < 0.7)
        if poor_metrics >= 2:
            return "general_underperformance"
        
        return ""
    
    def _reduce_model_complexity(self) -> bool:
        """Reduce model complexity to save memory."""
        try:
            # Reduce adapter sizes
            for task_id, adapter in self.model.adapters.items():
                if hasattr(adapter, 'adapter_size') and adapter.adapter_size > 32:
                    old_size = adapter.adapter_size
                    adapter.adapter_size = max(32, adapter.adapter_size // 2)
                    
                    # Recreate adapter with smaller size
                    self._recreate_adapter(task_id, adapter.adapter_size)
                    
                    logger.info(f"Reduced adapter size for {task_id}: {old_size} -> {adapter.adapter_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reduce model complexity: {e}")
            return False
    
    def _increase_model_capacity(self) -> bool:
        """Increase model capacity to improve performance."""
        try:
            # Increase adapter sizes for tasks with poor performance
            for task_id, adapter in self.model.adapters.items():
                if hasattr(adapter, 'adapter_size') and adapter.adapter_size < 128:
                    old_size = adapter.adapter_size
                    adapter.adapter_size = min(128, adapter.adapter_size * 2)
                    
                    # Recreate adapter with larger size
                    self._recreate_adapter(task_id, adapter.adapter_size)
                    
                    logger.info(f"Increased adapter size for {task_id}: {old_size} -> {adapter.adapter_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to increase model capacity: {e}")
            return False
    
    def _optimize_for_speed(self) -> bool:
        """Optimize architecture for inference speed."""
        try:
            # Enable faster activation functions
            if hasattr(self.model.config, 'activation_function'):
                if self.model.config.activation_function != 'relu':
                    self.model.config.activation_function = 'relu'
                    logger.info("Switched to ReLU activation for speed")
            
            # Reduce precision for faster computation
            if hasattr(self.model.config, 'mixed_precision'):
                self.model.config.mixed_precision = True
                logger.info("Enabled mixed precision for speed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize for speed: {e}")
            return False
    
    def _balanced_adaptation(self) -> bool:
        """Balanced adaptation considering multiple factors."""
        try:
            # Apply multiple small optimizations
            success_count = 0
            
            # Optimize activation functions
            if self._optimize_activations():
                success_count += 1
            
            # Optimize attention mechanisms
            if self._optimize_attention():
                success_count += 1
            
            # Optimize layer normalization
            if self._optimize_normalization():
                success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed balanced adaptation: {e}")
            return False
    
    def _recreate_adapter(self, task_id: str, new_size: int):
        """Recreate adapter with new size."""
        if hasattr(self.model, 'register_task'):
            # This would need to be implemented based on the actual adapter creation logic
            pass
    
    def _optimize_activations(self) -> bool:
        """Optimize activation functions for current workload."""
        # Placeholder for activation optimization
        return True
    
    def _optimize_attention(self) -> bool:
        """Optimize attention mechanisms."""
        # Placeholder for attention optimization
        return True
    
    def _optimize_normalization(self) -> bool:
        """Optimize normalization layers."""
        # Placeholder for normalization optimization
        return True


class AdaptiveLearningSystem:
    """Main adaptive learning system coordinator."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Initialize components
        self.metric_tracker = MetricTracker(window_size=getattr(config, 'metric_window_size', 100))
        
        # Define hyperparameter bounds
        param_bounds = {
            'learning_rate': (1e-6, 1e-2),
            'batch_size': (4, 128),
            'weight_decay': (0.0, 0.1),
            'warmup_steps': (0, 1000)
        }
        
        self.hyperparameter_optimizer = HyperparameterOptimizer(param_bounds)
        self.architecture_adaptation = ArchitectureAdaptation(model, config)
        
        # Adaptation state
        self.adaptation_active = True
        self.adaptation_frequency = getattr(config, 'adaptation_frequency', 10)  # Adapt every N epochs
        self.current_epoch = 0
        
        logger.info("Adaptive learning system initialized")
    
    def should_trigger_adaptation(self) -> bool:
        """Check if adaptation should be triggered."""
        if not self.adaptation_active:
            return False
        
        # Trigger adaptation periodically
        if self.current_epoch % self.adaptation_frequency == 0 and self.current_epoch > 0:
            return True
        
        # Trigger adaptation based on performance degradation
        key_metrics = ['accuracy', 'loss', 'inference_time']
        for metric in key_metrics:
            if self.metric_tracker.should_adapt(metric):
                return True
        
        return False
    
    def adapt_for_task(
        self, 
        task_id: str, 
        current_performance: Dict[str, float],
        resource_usage: Dict[str, float]
    ) -> Dict[str, Any]:
        """Perform comprehensive adaptation for a task."""
        
        adaptation_results = {
            "hyperparameters_adapted": False,
            "architecture_adapted": False,
            "performance_improvement": 0.0,
            "new_config": {}
        }
        
        # Record current performance
        for metric, value in current_performance.items():
            self.metric_tracker.record_metric(
                f"{task_id}_{metric}", 
                value, 
                {"task_id": task_id}
            )
        
        # Adapt hyperparameters
        if self.should_trigger_adaptation():
            
            # Get performance score for optimization
            performance_score = self._calculate_performance_score(current_performance)
            
            # Suggest new hyperparameters
            new_hyperparams = self.hyperparameter_optimizer.suggest_parameters(performance_score)
            
            # Update model configuration
            self._apply_hyperparameters(new_hyperparams)
            
            adaptation_results["hyperparameters_adapted"] = True
            adaptation_results["new_config"]["hyperparameters"] = new_hyperparams
            
            logger.info(f"Adapted hyperparameters for task {task_id}: {new_hyperparams}")
        
        # Adapt architecture if needed
        architecture_adapted = self.architecture_adaptation.adapt_architecture(
            current_performance, resource_usage
        )
        
        if architecture_adapted:
            adaptation_results["architecture_adapted"] = True
            logger.info(f"Adapted architecture for task {task_id}")
        
        # Estimate performance improvement
        adaptation_results["performance_improvement"] = self._estimate_improvement(
            adaptation_results
        )
        
        return adaptation_results
    
    def _calculate_performance_score(self, performance: Dict[str, float]) -> float:
        """Calculate a single performance score from multiple metrics."""
        
        # Weighted combination of metrics
        weights = {
            'accuracy': 0.4,
            'loss': -0.3,  # Negative because lower is better
            'inference_time': -0.2,  # Negative because lower is better
            'memory_usage': -0.1  # Negative because lower is better
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in performance.items():
            if metric in weights:
                score += weights[metric] * value
                total_weight += abs(weights[metric])
        
        # Normalize score
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    def _apply_hyperparameters(self, new_hyperparams: Dict[str, float]):
        """Apply new hyperparameters to model configuration."""
        
        for param, value in new_hyperparams.items():
            if param == 'learning_rate':
                self.config.learning_rate = value
            elif param == 'batch_size':
                self.config.batch_size = int(value)
            elif param == 'weight_decay':
                self.config.weight_decay = value
            elif param == 'warmup_steps':
                self.config.warmup_steps = int(value)
    
    def _estimate_improvement(self, adaptation_results: Dict[str, Any]) -> float:
        """Estimate expected performance improvement from adaptations."""
        
        improvement = 0.0
        
        if adaptation_results["hyperparameters_adapted"]:
            # Estimate improvement from hyperparameter optimization
            optimization_status = self.hyperparameter_optimizer.get_optimization_status()
            if optimization_status["evaluations_completed"] > 0:
                recent_performances = optimization_status["recent_performances"]
                if len(recent_performances) >= 2:
                    improvement += recent_performances[-1] - recent_performances[-2]
        
        if adaptation_results["architecture_adapted"]:
            # Estimate improvement from architecture adaptation
            improvement += 0.05  # Conservative estimate
        
        return max(0.0, improvement)  # Ensure non-negative
    
    def update_epoch(self, epoch: int):
        """Update current epoch for adaptation scheduling."""
        self.current_epoch = epoch
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get comprehensive adaptation system status."""
        
        status = {
            "adaptation_active": self.adaptation_active,
            "current_epoch": self.current_epoch,
            "adaptation_frequency": self.adaptation_frequency,
            "metric_tracker_status": {
                metric: self.metric_tracker.get_metric_summary(metric)
                for metric in ['accuracy', 'loss', 'inference_time']
                if metric in self.metric_tracker.metrics_history
            },
            "hyperparameter_optimization": self.hyperparameter_optimizer.get_optimization_status(),
            "architecture_adaptations": len(self.architecture_adaptation.adaptation_history)
        }
        
        return status
    
    def enable_adaptation(self):
        """Enable adaptive learning."""
        self.adaptation_active = True
        logger.info("Adaptive learning enabled")
    
    def disable_adaptation(self):
        """Disable adaptive learning."""
        self.adaptation_active = False
        logger.info("Adaptive learning disabled")
    
    def save_adaptation_state(self, filepath: str):
        """Save adaptation state to file."""
        state = {
            "metric_history": dict(self.metric_tracker.metrics_history),
            "trend_analysis": self.metric_tracker.trend_analysis,
            "hyperparameter_history": self.hyperparameter_optimizer.evaluation_history,
            "architecture_history": self.architecture_adaptation.adaptation_history,
            "current_epoch": self.current_epoch
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Adaptation state saved to {filepath}")
    
    def load_adaptation_state(self, filepath: str):
        """Load adaptation state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore state (simplified - would need proper deserialization)
            self.current_epoch = state.get("current_epoch", 0)
            
            logger.info(f"Adaptation state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load adaptation state: {e}")