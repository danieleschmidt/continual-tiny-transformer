"""Task management system for continual learning."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQUENCE_LABELING = "sequence_labeling"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"


class TaskStatus(Enum):
    """Task learning status."""
    REGISTERED = "registered"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Task definition and metadata."""
    task_id: str
    task_type: TaskType
    num_labels: int
    description: str = ""
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.REGISTERED
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Performance tracking
    best_accuracy: float = 0.0
    best_loss: float = float('inf')
    training_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Memory tracking
    parameter_count: int = 0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "num_labels": self.num_labels,
            "description": self.description,
            "dataset_info": self.dataset_info,
            "config": self.config,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "best_accuracy": self.best_accuracy,
            "best_loss": self.best_loss,
            "training_history": self.training_history,
            "parameter_count": self.parameter_count,
            "memory_usage_mb": self.memory_usage_mb
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        # Convert datetime strings back to datetime objects
        created_at = datetime.fromisoformat(data["created_at"])
        completed_at = datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None
        
        return cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            num_labels=data["num_labels"],
            description=data.get("description", ""),
            dataset_info=data.get("dataset_info", {}),
            config=data.get("config", {}),
            status=TaskStatus(data["status"]),
            created_at=created_at,
            completed_at=completed_at,
            best_accuracy=data.get("best_accuracy", 0.0),
            best_loss=data.get("best_loss", float('inf')),
            training_history=data.get("training_history", []),
            parameter_count=data.get("parameter_count", 0),
            memory_usage_mb=data.get("memory_usage_mb", 0.0)
        )


class TaskManager:
    """Manages tasks in continual learning scenario."""
    
    def __init__(self, config):
        self.config = config
        self.tasks: Dict[str, Task] = {}
        self.task_order: List[str] = []
        self.current_task_id: Optional[str] = None
        
        # Task relationships
        self.task_dependencies: Dict[str, Set[str]] = {}  # task_id -> prerequisite task_ids
        self.task_similarities: Dict[str, Dict[str, float]] = {}  # task similarity matrix
        
        # Performance tracking
        self.forgetting_matrix: Dict[str, Dict[str, float]] = {}  # forgetting between tasks
        self.transfer_matrix: Dict[str, Dict[str, float]] = {}  # transfer learning effects
        
        # Load existing tasks if available
        self.load_tasks()
    
    def add_task(
        self,
        task_id: str,
        task_type: str,
        num_labels: int,
        description: str = "",
        dataset_info: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        prerequisites: Optional[List[str]] = None
    ) -> Task:
        """Add a new task to the manager."""
        
        if task_id in self.tasks:
            logger.warning(f"Task '{task_id}' already exists")
            return self.tasks[task_id]
        
        # Validate task type
        try:
            task_type_enum = TaskType(task_type)
        except ValueError:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Check if we've reached task limit
        if len(self.tasks) >= self.config.max_tasks:
            raise ValueError(f"Maximum number of tasks ({self.config.max_tasks}) exceeded")
        
        # Validate prerequisites
        if prerequisites:
            for prereq in prerequisites:
                if prereq not in self.tasks:
                    raise ValueError(f"Prerequisite task '{prereq}' not found")
        
        # Create task
        task = Task(
            task_id=task_id,
            task_type=task_type_enum,
            num_labels=num_labels,
            description=description,
            dataset_info=dataset_info or {},
            config=config or {}
        )
        
        # Add to manager
        self.tasks[task_id] = task
        self.task_order.append(task_id)
        
        # Set up dependencies
        if prerequisites:
            self.task_dependencies[task_id] = set(prerequisites)
        
        # Initialize performance tracking
        self.forgetting_matrix[task_id] = {}
        self.transfer_matrix[task_id] = {}
        self.task_similarities[task_id] = {}
        
        logger.info(f"Added task '{task_id}' (type: {task_type}, labels: {num_labels})")
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in order."""
        return [self.tasks[task_id] for task_id in self.task_order]
    
    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]
    
    def get_learning_order(self) -> List[str]:
        """Get optimal task learning order based on dependencies and similarities."""
        # Start with topological sort based on dependencies
        order = self._topological_sort()
        
        # Optimize order based on task similarities (future enhancement)
        # For now, return dependency-based order
        return order
    
    def _topological_sort(self) -> List[str]:
        """Topological sort of tasks based on dependencies."""
        # Kahn's algorithm for topological sorting
        in_degree = {task_id: 0 for task_id in self.tasks}
        
        # Calculate in-degrees
        for task_id, deps in self.task_dependencies.items():
            in_degree[task_id] = len(deps)
        
        # Find tasks with no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by task order for deterministic results
            queue.sort(key=lambda x: self.task_order.index(x))
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent tasks
            for task_id, deps in self.task_dependencies.items():
                if current in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        # Check for cycles
        if len(result) != len(self.tasks):
            raise ValueError("Circular dependency detected in tasks")
        
        return result
    
    def set_current_task(self, task_id: str):
        """Set the current active task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        self.current_task_id = task_id
        self.tasks[task_id].status = TaskStatus.LEARNING
        logger.info(f"Set current task to '{task_id}'")
    
    def complete_task(self, task_id: str, final_metrics: Optional[Dict[str, float]] = None):
        """Mark a task as completed."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        
        if final_metrics:
            task.best_accuracy = final_metrics.get("accuracy", task.best_accuracy)
            task.best_loss = final_metrics.get("loss", task.best_loss)
            task.training_history.append(final_metrics)
        
        logger.info(f"Completed task '{task_id}'")
        
        # Auto-save after task completion
        self.save_tasks()
    
    def fail_task(self, task_id: str, error_message: str = ""):
        """Mark a task as failed."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.config["error_message"] = error_message
        
        logger.error(f"Task '{task_id}' failed: {error_message}")
    
    def update_task_performance(
        self,
        task_id: str,
        metrics: Dict[str, float],
        memory_usage_mb: Optional[float] = None,
        parameter_count: Optional[int] = None
    ):
        """Update task performance metrics."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        task = self.tasks[task_id]
        
        # Update best metrics
        if "accuracy" in metrics and metrics["accuracy"] > task.best_accuracy:
            task.best_accuracy = metrics["accuracy"]
        
        if "loss" in metrics and metrics["loss"] < task.best_loss:
            task.best_loss = metrics["loss"]
        
        # Add to training history
        task.training_history.append(metrics.copy())
        
        # Update resource usage
        if memory_usage_mb is not None:
            task.memory_usage_mb = memory_usage_mb
        
        if parameter_count is not None:
            task.parameter_count = parameter_count
    
    def compute_forgetting(self, old_task_id: str, new_task_id: str, old_performance: float, new_performance: float):
        """Compute and store forgetting metric between tasks."""
        forgetting = max(0, old_performance - new_performance)
        
        if old_task_id not in self.forgetting_matrix:
            self.forgetting_matrix[old_task_id] = {}
        
        self.forgetting_matrix[old_task_id][new_task_id] = forgetting
        logger.info(f"Forgetting from {old_task_id} to {new_task_id}: {forgetting:.4f}")
    
    def compute_transfer(self, source_task_id: str, target_task_id: str, baseline_performance: float, actual_performance: float):
        """Compute and store transfer learning effect."""
        transfer = actual_performance - baseline_performance
        
        if source_task_id not in self.transfer_matrix:
            self.transfer_matrix[source_task_id] = {}
        
        self.transfer_matrix[source_task_id][target_task_id] = transfer
        logger.info(f"Transfer from {source_task_id} to {target_task_id}: {transfer:.4f}")
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task statistics."""
        completed_tasks = self.get_completed_tasks()
        
        if not completed_tasks:
            return {"message": "No completed tasks"}
        
        # Basic statistics
        total_tasks = len(self.tasks)
        completed_count = len(completed_tasks)
        
        # Performance statistics
        accuracies = [task.best_accuracy for task in completed_tasks]
        losses = [task.best_loss for task in completed_tasks if task.best_loss != float('inf')]
        
        # Memory statistics
        memory_usage = [task.memory_usage_mb for task in completed_tasks if task.memory_usage_mb > 0]
        parameter_counts = [task.parameter_count for task in completed_tasks if task.parameter_count > 0]
        
        # Forgetting statistics
        all_forgetting = []
        for task_forgetting in self.forgetting_matrix.values():
            all_forgetting.extend(task_forgetting.values())
        
        # Transfer statistics
        all_transfer = []
        for task_transfer in self.transfer_matrix.values():
            all_transfer.extend(task_transfer.values())
        
        stats = {
            "total_tasks": total_tasks,
            "completed_tasks": completed_count,
            "completion_rate": completed_count / total_tasks if total_tasks > 0 else 0,
            
            "performance": {
                "mean_accuracy": np.mean(accuracies) if accuracies else 0,
                "std_accuracy": np.std(accuracies) if accuracies else 0,
                "min_accuracy": min(accuracies) if accuracies else 0,
                "max_accuracy": max(accuracies) if accuracies else 0,
                
                "mean_loss": np.mean(losses) if losses else 0,
                "std_loss": np.std(losses) if losses else 0,
            },
            
            "memory": {
                "mean_memory_mb": np.mean(memory_usage) if memory_usage else 0,
                "total_memory_mb": sum(memory_usage) if memory_usage else 0,
                "mean_parameters": np.mean(parameter_counts) if parameter_counts else 0,
                "total_parameters": sum(parameter_counts) if parameter_counts else 0,
            },
            
            "continual_learning": {
                "mean_forgetting": np.mean(all_forgetting) if all_forgetting else 0,
                "max_forgetting": max(all_forgetting) if all_forgetting else 0,
                "mean_transfer": np.mean(all_transfer) if all_transfer else 0,
                "positive_transfer_ratio": sum(1 for t in all_transfer if t > 0) / len(all_transfer) if all_transfer else 0,
            }
        }
        
        return stats
    
    def get_task_summary(self) -> str:
        """Get human-readable task summary."""
        stats = self.get_task_statistics()
        
        if "message" in stats:
            return stats["message"]
        
        summary = f"""
Task Summary:
=============
Total Tasks: {stats['total_tasks']}
Completed: {stats['completed_tasks']} ({stats['completion_rate']:.1%})

Performance:
- Mean Accuracy: {stats['performance']['mean_accuracy']:.3f} ± {stats['performance']['std_accuracy']:.3f}
- Best Accuracy: {stats['performance']['max_accuracy']:.3f}
- Worst Accuracy: {stats['performance']['min_accuracy']:.3f}

Memory Efficiency:
- Total Parameters: {stats['memory']['total_parameters']:,}
- Mean Memory per Task: {stats['memory']['mean_memory_mb']:.1f} MB

Continual Learning:
- Mean Forgetting: {stats['continual_learning']['mean_forgetting']:.3f}
- Mean Transfer: {stats['continual_learning']['mean_transfer']:.3f}
- Positive Transfer Rate: {stats['continual_learning']['positive_transfer_ratio']:.1%}
        """.strip()
        
        return summary
    
    def save_tasks(self, filepath: Optional[str] = None):
        """Save task information to file."""
        if filepath is None:
            filepath = Path(self.config.output_dir) / "tasks.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tasks to dictionaries
        tasks_data = {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "task_order": self.task_order,
            "current_task_id": self.current_task_id,
            "task_dependencies": {k: list(v) for k, v in self.task_dependencies.items()},
            "forgetting_matrix": self.forgetting_matrix,
            "transfer_matrix": self.transfer_matrix,
            "task_similarities": self.task_similarities,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(tasks_data, f, indent=2)
        
        logger.info(f"Saved {len(self.tasks)} tasks to {filepath}")
    
    def load_tasks(self, filepath: Optional[str] = None):
        """Load task information from file."""
        if filepath is None:
            filepath = Path(self.config.output_dir) / "tasks.json"
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.info("No existing tasks file found")
            return
        
        try:
            with open(filepath, 'r') as f:
                tasks_data = json.load(f)
            
            # Restore tasks
            self.tasks = {}
            for task_id, task_dict in tasks_data["tasks"].items():
                self.tasks[task_id] = Task.from_dict(task_dict)
            
            # Restore other data
            self.task_order = tasks_data.get("task_order", list(self.tasks.keys()))
            self.current_task_id = tasks_data.get("current_task_id")
            
            # Restore dependencies (convert lists back to sets)
            self.task_dependencies = {
                k: set(v) for k, v in tasks_data.get("task_dependencies", {}).items()
            }
            
            self.forgetting_matrix = tasks_data.get("forgetting_matrix", {})
            self.transfer_matrix = tasks_data.get("transfer_matrix", {})
            self.task_similarities = tasks_data.get("task_similarities", {})
            
            logger.info(f"Loaded {len(self.tasks)} tasks from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load tasks from {filepath}: {e}")
    
    def reset(self):
        """Reset all task information."""
        self.tasks.clear()
        self.task_order.clear()
        self.current_task_id = None
        self.task_dependencies.clear()
        self.task_similarities.clear()
        self.forgetting_matrix.clear()
        self.transfer_matrix.clear()
        
        logger.info("Reset all task information")
    
    def __len__(self) -> int:
        """Number of registered tasks."""
        return len(self.tasks)
    
    def __contains__(self, task_id: str) -> bool:
        """Check if task is registered."""
        return task_id in self.tasks
    
    def __iter__(self):
        """Iterate over tasks in order."""
        for task_id in self.task_order:
            yield self.tasks[task_id]


# Import numpy for statistics (with fallback)
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available, using Python statistics")
    
    class np:
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0
        
        @staticmethod
        def std(x):
            if not x:
                return 0
            mean_val = sum(x) / len(x)
            return (sum((xi - mean_val) ** 2 for xi in x) / len(x)) ** 0.5