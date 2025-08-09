"""High-level API for continual learning."""

from typing import Any, Dict, List, Optional, Union
import torch
from pathlib import Path

from ..core import ContinualTransformer, ContinualConfig
from ..data.loaders import create_dataloader
from ..metrics.continual_metrics import ContinualMetrics

class ContinualLearningAPI:
    """High-level API for continual learning workflows."""
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        max_tasks: int = 50,
        device: Optional[str] = None
    ):
        """Initialize the continual learning API.
        
        Args:
            model_name: Base transformer model name
            max_tasks: Maximum number of tasks to support
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.config = ContinualConfig(
            model_name=model_name,
            max_tasks=max_tasks,
            device=device
        )
        
        self.model = ContinualTransformer(self.config)
        self.metrics = ContinualMetrics()
        self.trained_tasks = set()
    
    def add_task(
        self, 
        task_id: str, 
        num_labels: int, 
        task_type: str = "classification"
    ):
        """Add a new task to the model.
        
        Args:
            task_id: Unique identifier for the task
            num_labels: Number of output labels for the task
            task_type: Type of task ('classification', 'regression', etc.)
        """
        self.model.register_task(task_id, num_labels, task_type)
        print(f"✅ Added task '{task_id}' with {num_labels} labels")
    
    def train_task(
        self,
        task_id: str,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> Dict[str, float]:
        """Train on a specific task.
        
        Args:
            task_id: Task identifier
            train_data: Training data as list of dicts with 'text' and 'label' keys
            eval_data: Evaluation data (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Training metrics dictionary
        """
        # Update config with training parameters
        self.config.num_epochs = epochs
        self.config.batch_size = batch_size
        self.config.learning_rate = learning_rate
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_data, 
            self.config, 
            shuffle=True
        )
        
        eval_loader = None
        if eval_data:
            eval_loader = create_dataloader(
                eval_data, 
                self.config, 
                shuffle=False
            )
        
        # Train the task
        print(f"🚀 Training task '{task_id}' for {epochs} epochs...")
        self.model.learn_task(
            task_id=task_id,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            num_epochs=epochs
        )
        
        self.trained_tasks.add(task_id)
        
        # Return final metrics
        performance = self.model.task_performance.get(task_id, {})
        final_metrics = {
            "train_accuracy": performance.get("train_accuracy", [])[-1] if performance.get("train_accuracy") else 0.0,
            "eval_accuracy": performance.get("eval_accuracy", [])[-1] if performance.get("eval_accuracy") else 0.0,
            "train_loss": performance.get("train_loss", [])[-1] if performance.get("train_loss") else 0.0,
            "eval_loss": performance.get("eval_loss", [])[-1] if performance.get("eval_loss") else 0.0,
        }
        
        print(f"✅ Training completed! Final accuracy: {final_metrics['train_accuracy']:.4f}")
        return final_metrics
    
    def predict(
        self, 
        text: Union[str, List[str]], 
        task_id: str
    ) -> Dict[str, Any]:
        """Make predictions for given text(s).
        
        Args:
            text: Input text or list of texts
            task_id: Task identifier
            
        Returns:
            Predictions with probabilities
        """
        if task_id not in self.trained_tasks:
            print(f"⚠️  Warning: Task '{task_id}' has not been trained yet")
        
        return self.model.predict(text, task_id)
    
    def evaluate_task(
        self, 
        task_id: str, 
        eval_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate performance on a task.
        
        Args:
            task_id: Task identifier
            eval_data: Evaluation data
            
        Returns:
            Evaluation metrics
        """
        eval_loader = create_dataloader(
            eval_data, 
            self.config, 
            shuffle=False
        )
        
        return self.model.evaluate_task(task_id, eval_loader)
    
    def evaluate_all_tasks(
        self, 
        eval_data_dict: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate performance on all trained tasks.
        
        Args:
            eval_data_dict: Dictionary mapping task_id to eval_data
            
        Returns:
            Nested dictionary of evaluation metrics per task
        """
        results = {}
        
        for task_id in self.trained_tasks:
            if task_id in eval_data_dict:
                print(f"🔍 Evaluating task '{task_id}'...")
                results[task_id] = self.evaluate_task(task_id, eval_data_dict[task_id])
            else:
                print(f"⚠️  No evaluation data provided for task '{task_id}'")
        
        # Compute continual learning metrics
        if len(results) > 1:
            cl_metrics = self.metrics.compute_continual_metrics(results)
            print(f"📊 Continual Learning Metrics:")
            print(f"   Average Accuracy: {cl_metrics.get('average_accuracy', 0):.4f}")
            print(f"   Forgetting Rate: {cl_metrics.get('forgetting_rate', 0):.4f}")
            print(f"   Knowledge Retention: {cl_metrics.get('knowledge_retention', 0):.4f}")
        
        return results
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        return self.model.get_memory_usage()
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about registered and trained tasks."""
        return {
            "registered_tasks": list(self.model.adapters.keys()),
            "trained_tasks": list(self.trained_tasks),
            "num_tasks": len(self.model.adapters),
            "max_tasks": self.config.max_tasks,
            "memory_usage": self.get_memory_usage()
        }
    
    def save(self, save_path: str):
        """Save the model and metadata.
        
        Args:
            save_path: Directory to save the model
        """
        self.model.save_model(save_path)
        
        # Save API metadata
        metadata = {
            "trained_tasks": list(self.trained_tasks),
            "api_version": "1.0.0"
        }
        
        with open(Path(save_path) / "api_metadata.json", "w") as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str, **kwargs):
        """Load a saved model.
        
        Args:
            load_path: Directory containing saved model
            **kwargs: Additional arguments for API initialization
            
        Returns:
            Loaded ContinualLearningAPI instance
        """
        # Load the model
        model = ContinualTransformer.load_model(load_path)
        
        # Create API instance
        api = cls.__new__(cls)
        api.model = model
        api.config = model.config
        api.metrics = ContinualMetrics()
        
        # Load API metadata
        metadata_path = Path(load_path) / "api_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            api.trained_tasks = set(metadata.get("trained_tasks", []))
        else:
            api.trained_tasks = set(model.adapters.keys())
        
        print(f"📂 Model loaded from {load_path}")
        return api
    
    def optimize_for_deployment(self, optimization_level: str = "balanced"):
        """Optimize model for deployment.
        
        Args:
            optimization_level: 'speed', 'memory', or 'balanced'
        """
        print(f"⚡ Optimizing model for {optimization_level} deployment...")
        optimizations = self.model.optimize_for_inference(optimization_level)
        print(f"✅ Applied optimizations: {list(optimizations.keys())}")
        return optimizations

__all__ = ["ContinualLearningAPI"]