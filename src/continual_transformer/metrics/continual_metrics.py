"""Metrics for evaluating continual learning performance."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    task_id: str
    accuracy: float = 0.0
    loss: float = float('inf')
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    num_samples: int = 0
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "num_samples": self.num_samples,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "memory_usage_mb": self.memory_usage_mb,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ContinualLearningMetrics:
    """Comprehensive metrics for continual learning evaluation."""
    
    # Task-specific metrics
    task_metrics: Dict[str, TaskMetrics] = field(default_factory=dict)
    
    # Continual learning specific metrics
    average_accuracy: float = 0.0
    backward_transfer: float = 0.0  # How much old tasks improve
    forward_transfer: float = 0.0   # How much new tasks benefit from old ones
    forgetting: float = 0.0         # How much performance drops on old tasks
    learning_accuracy: float = 0.0  # Final accuracy on each task when first learned
    
    # Memory and efficiency metrics
    total_parameters: int = 0
    memory_growth_rate: float = 0.0
    average_training_time: float = 0.0
    parameter_efficiency: float = 0.0  # accuracy per parameter
    
    # Task ordering and learning trajectory
    task_order: List[str] = field(default_factory=list)
    learning_curve: List[float] = field(default_factory=list)  # Average accuracy over time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_metrics": {k: v.to_dict() for k, v in self.task_metrics.items()},
            "average_accuracy": self.average_accuracy,
            "backward_transfer": self.backward_transfer,
            "forward_transfer": self.forward_transfer,
            "forgetting": self.forgetting,
            "learning_accuracy": self.learning_accuracy,
            "total_parameters": self.total_parameters,
            "memory_growth_rate": self.memory_growth_rate,
            "average_training_time": self.average_training_time,
            "parameter_efficiency": self.parameter_efficiency,
            "task_order": self.task_order,
            "learning_curve": self.learning_curve
        }


class ContinualMetrics:
    """Comprehensive metrics computation for continual learning."""
    
    def __init__(self):
        # Store performance matrix: performance[task_learned][task_evaluated]
        self.performance_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Store baseline performance (without continual learning)
        self.baseline_performance: Dict[str, float] = {}
        
        # Store final performance when each task was first learned
        self.initial_performance: Dict[str, float] = {}
        
        # Store training metadata
        self.training_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Task ordering
        self.task_sequence: List[str] = []
        
        # Memory tracking
        self.parameter_counts: List[int] = []
        
    def record_task_performance(
        self,
        current_task: str,
        evaluated_task: str,
        accuracy: float,
        loss: float = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Record performance of evaluated_task after learning current_task."""
        self.performance_matrix[current_task][evaluated_task] = accuracy
        
        # Record initial performance when task is first learned
        if current_task == evaluated_task and evaluated_task not in self.initial_performance:
            self.initial_performance[evaluated_task] = accuracy
        
        # Store additional metrics
        if additional_metrics:
            if current_task not in self.training_metadata:
                self.training_metadata[current_task] = {}
            self.training_metadata[current_task][f"{evaluated_task}_metrics"] = additional_metrics
        
        logger.debug(f"Recorded {evaluated_task} accuracy: {accuracy:.4f} after learning {current_task}")
    
    def record_baseline_performance(self, task_id: str, accuracy: float):
        """Record baseline performance for a task (trained in isolation)."""
        self.baseline_performance[task_id] = accuracy
        logger.debug(f"Recorded baseline {task_id} accuracy: {accuracy:.4f}")
    
    def add_task_to_sequence(self, task_id: str):
        """Add task to the learning sequence."""
        if task_id not in self.task_sequence:
            self.task_sequence.append(task_id)
    
    def record_parameter_count(self, count: int):
        """Record parameter count at current stage."""
        self.parameter_counts.append(count)
    
    def compute_average_accuracy(self) -> float:
        """Compute average accuracy across all tasks at the end of learning."""
        if not self.task_sequence:
            return 0.0
        
        final_task = self.task_sequence[-1]
        if final_task not in self.performance_matrix:
            return 0.0
        
        accuracies = []
        for task in self.task_sequence:
            if task in self.performance_matrix[final_task]:
                accuracies.append(self.performance_matrix[final_task][task])
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def compute_forgetting(self) -> float:
        """Compute average forgetting across all tasks.
        
        Forgetting = max_j≤i R_{i,j} - R_{T,j}
        where R_{i,j} is performance on task j after learning task i
        """
        if len(self.task_sequence) < 2:
            return 0.0
        
        forgetting_values = []
        
        for j, task_j in enumerate(self.task_sequence[:-1]):  # Exclude last task
            max_accuracy = 0.0
            
            # Find maximum accuracy on task j across all later learning stages
            for i, task_i in enumerate(self.task_sequence[j:], j):
                if task_i in self.performance_matrix and task_j in self.performance_matrix[task_i]:
                    max_accuracy = max(max_accuracy, self.performance_matrix[task_i][task_j])
            
            # Final accuracy on task j
            final_task = self.task_sequence[-1]
            if final_task in self.performance_matrix and task_j in self.performance_matrix[final_task]:
                final_accuracy = self.performance_matrix[final_task][task_j]
                forgetting = max(0, max_accuracy - final_accuracy)
                forgetting_values.append(forgetting)
        
        return np.mean(forgetting_values) if forgetting_values else 0.0
    
    def compute_backward_transfer(self) -> float:
        """Compute backward transfer (how learning new tasks helps old tasks).
        
        BWT = 1/(T-1) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
        """
        if len(self.task_sequence) < 2:
            return 0.0
        
        transfer_values = []
        final_task = self.task_sequence[-1]
        
        for i, task_i in enumerate(self.task_sequence[:-1]):
            # Performance when task was initially learned vs. final performance
            initial_perf = self.initial_performance.get(task_i, 0.0)
            final_perf = self.performance_matrix.get(final_task, {}).get(task_i, 0.0)
            
            transfer = final_perf - initial_perf
            transfer_values.append(transfer)
        
        return np.mean(transfer_values) if transfer_values else 0.0
    
    def compute_forward_transfer(self) -> float:
        """Compute forward transfer (how old tasks help new tasks).
        
        FWT = 1/(T-1) * sum_{i=2}^{T} (b_{i,i} - b_{i,i}^*)
        where b_{i,i}^* is performance of task i trained from scratch
        """
        if len(self.task_sequence) < 2:
            return 0.0
        
        transfer_values = []
        
        for i, task_i in enumerate(self.task_sequence[1:], 1):  # Start from second task
            # Performance when first learning task i in continual setting
            initial_perf = self.initial_performance.get(task_i, 0.0)
            
            # Baseline performance (trained from scratch)
            baseline_perf = self.baseline_performance.get(task_i, 0.0)
            
            if baseline_perf > 0:  # Only compute if we have baseline
                transfer = initial_perf - baseline_perf
                transfer_values.append(transfer)
        
        return np.mean(transfer_values) if transfer_values else 0.0
    
    def compute_learning_accuracy(self) -> float:
        """Compute average accuracy when tasks were first learned."""
        if not self.initial_performance:
            return 0.0
        
        return np.mean(list(self.initial_performance.values()))
    
    def compute_memory_growth_rate(self) -> float:
        """Compute rate of parameter growth."""
        if len(self.parameter_counts) < 2:
            return 0.0
        
        initial_params = self.parameter_counts[0]
        final_params = self.parameter_counts[-1]
        
        if initial_params == 0:
            return float('inf') if final_params > 0 else 0.0
        
        growth_rate = (final_params - initial_params) / initial_params
        return growth_rate
    
    def compute_parameter_efficiency(self) -> float:
        """Compute accuracy per parameter."""
        avg_accuracy = self.compute_average_accuracy()
        total_params = self.parameter_counts[-1] if self.parameter_counts else 1
        
        return avg_accuracy / max(total_params, 1) * 1e6  # Accuracy per million parameters
    
    def compute_all_metrics(self) -> ContinualLearningMetrics:
        """Compute all continual learning metrics."""
        metrics = ContinualLearningMetrics()
        
        # Basic metrics
        metrics.average_accuracy = self.compute_average_accuracy()
        metrics.backward_transfer = self.compute_backward_transfer()
        metrics.forward_transfer = self.compute_forward_transfer()
        metrics.forgetting = self.compute_forgetting()
        metrics.learning_accuracy = self.compute_learning_accuracy()
        
        # Memory and efficiency
        metrics.total_parameters = self.parameter_counts[-1] if self.parameter_counts else 0
        metrics.memory_growth_rate = self.compute_memory_growth_rate()
        metrics.parameter_efficiency = self.compute_parameter_efficiency()      
        
        # Task sequence
        metrics.task_order = self.task_sequence.copy()
        
        # Learning curve (average accuracy after each task)
        learning_curve = []
        for i, task in enumerate(self.task_sequence):
            if task in self.performance_matrix:
                # Average accuracy on all tasks learned so far
                accuracies = []
                for prev_task in self.task_sequence[:i+1]:
                    if prev_task in self.performance_matrix[task]:
                        accuracies.append(self.performance_matrix[task][prev_task])
                
                avg_acc = np.mean(accuracies) if accuracies else 0.0
                learning_curve.append(avg_acc)
        
        metrics.learning_curve = learning_curve
        
        # Individual task metrics
        for task_id in self.task_sequence:
            task_metrics = TaskMetrics(task_id=task_id)
            
            # Get final performance
            final_task = self.task_sequence[-1]
            if final_task in self.performance_matrix and task_id in self.performance_matrix[final_task]:
                task_metrics.accuracy = self.performance_matrix[final_task][task_id]
            
            # Get additional metrics if available
            if task_id in self.training_metadata:
                metadata = self.training_metadata[task_id]
                task_metrics.training_time = metadata.get('training_time', 0.0)
                task_metrics.memory_usage_mb = metadata.get('memory_usage_mb', 0.0)
                task_metrics.num_samples = metadata.get('num_samples', 0)
            
            metrics.task_metrics[task_id] = task_metrics
        
        return metrics
    
    def get_performance_matrix_df(self):
        """Get performance matrix as pandas DataFrame (if available)."""
        try:
            import pandas as pd
            
            # Create matrix with tasks as both rows and columns
            all_tasks = sorted(set(self.task_sequence))
            matrix_data = []
            
            for learned_task in all_tasks:
                row = []
                for eval_task in all_tasks:
                    perf = self.performance_matrix.get(learned_task, {}).get(eval_task, np.nan)
                    row.append(perf)
                matrix_data.append(row)
            
            df = pd.DataFrame(
                matrix_data,
                index=[f"After_{t}" for t in all_tasks],
                columns=[f"Task_{t}" for t in all_tasks]
            )
            
            return df
            
        except ImportError:
            logger.warning("Pandas not available for DataFrame export")
            return None
    
    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        metrics = self.compute_all_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        logger.info(f"Saved continual learning metrics to {filepath}")
    
    def print_summary(self):
        """Print a summary of continual learning performance."""
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*60)
        print("CONTINUAL LEARNING METRICS SUMMARY")
        print("="*60)
        
        print(f"Tasks Learned: {len(self.task_sequence)}")
        print(f"Task Sequence: {' → '.join(self.task_sequence)}")
        print()
        
        print("PERFORMANCE METRICS:")
        print(f"  Average Accuracy:     {metrics.average_accuracy:.3f}")
        print(f"  Learning Accuracy:    {metrics.learning_accuracy:.3f}")
        print(f"  Forgetting:          {metrics.forgetting:.3f}")
        print(f"  Backward Transfer:   {metrics.backward_transfer:.3f}")
        print(f"  Forward Transfer:    {metrics.forward_transfer:.3f}")
        print()
        
        print("EFFICIENCY METRICS:")
        print(f"  Total Parameters:    {metrics.total_parameters:,}")
        print(f"  Memory Growth Rate:  {metrics.memory_growth_rate:.2%}")
        print(f"  Parameter Efficiency: {metrics.parameter_efficiency:.2f} acc/M params")
        print()
        
        print("TASK-SPECIFIC PERFORMANCE:")
        for task_id, task_metrics in metrics.task_metrics.items():
            print(f"  {task_id}: {task_metrics.accuracy:.3f}")
        
        print("="*60)
    
    def reset(self):
        """Reset all metrics."""
        self.performance_matrix.clear()
        self.baseline_performance.clear()
        self.initial_performance.clear()
        self.training_metadata.clear()
        self.task_sequence.clear()
        self.parameter_counts.clear()
        
        logger.info("Reset all continual learning metrics")