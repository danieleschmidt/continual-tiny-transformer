"""
Simplified API for Continual Tiny Transformer.
Provides easy-to-use interfaces for common continual learning tasks.
"""

import torch
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json

from ..core.model import ContinualTransformer
from ..core.config import ContinualConfig
from ..data.simple_datasets import (
    get_quick_demo_datasets, 
    create_dataloader_from_dataset,
    SyntheticDataGenerator
)
from ..metrics.continual_metrics import ContinualMetrics

logger = logging.getLogger(__name__)


class SimpleContinualTransformer:
    """Simplified interface for continual learning with transformers."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_tasks: int = 10,
        device: str = "auto",
        enable_optimizations: bool = True
    ):
        """
        Initialize simplified continual transformer.
        
        Args:
            model_name: HuggingFace model name or path
            max_tasks: Maximum number of tasks to support
            device: Device to use ("auto", "cpu", "cuda")
            enable_optimizations: Whether to enable performance optimizations
        """
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create configuration
        self.config = ContinualConfig(
            model_name=model_name,
            max_tasks=max_tasks,
            device=torch.device(device),
            freeze_base_model=True,
            adaptation_method="activation",
            use_knowledge_distillation=True,
            elastic_weight_consolidation=True,
            learning_rate=2e-5,
            num_epochs=5,
            batch_size=16,
            enable_monitoring=enable_optimizations,
            mixed_precision=torch.cuda.is_available() and enable_optimizations,
            gradient_clipping=1.0
        )
        
        # Initialize model
        self.model = ContinualTransformer(self.config)
        
        # Initialize metrics tracker
        self.metrics = ContinualMetrics()
        
        # Keep track of learned tasks
        self.learned_tasks = []
        
        logger.info(f"Initialized SimpleContinualTransformer with {model_name} on {device}")
    
    def add_task(
        self,
        task_name: str,
        num_classes: int,
        task_type: str = "classification"
    ) -> None:
        """
        Add a new task to the model.
        
        Args:
            task_name: Unique identifier for the task
            num_classes: Number of output classes for the task
            task_type: Type of task ("classification", "regression")
        """
        
        if task_name in self.learned_tasks:
            logger.warning(f"Task '{task_name}' already exists")
            return
        
        # Register task with model
        self.model.register_task(task_name, num_classes, task_type)
        self.metrics.add_task_to_sequence(task_name)
        
        logger.info(f"Added task '{task_name}' with {num_classes} classes")
    
    def learn_from_texts(
        self,
        task_name: str,
        texts: List[str],
        labels: List[int],
        validation_split: float = 0.2,
        epochs: int = None,
        batch_size: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Learn a task from text data.
        
        Args:
            task_name: Name of the task to learn
            texts: List of input texts
            labels: List of corresponding labels
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs (defaults to config)
            batch_size: Batch size (defaults to config)
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary with training results and metrics
        """
        
        if task_name not in [task for task in self.model.adapters.keys()]:
            raise ValueError(f"Task '{task_name}' not registered. Call add_task() first.")
        
        # Prepare data
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Secure default padding
        
        # Split data
        num_samples = len(texts)
        num_val = int(num_samples * validation_split)
        
        # Shuffle data
        import random
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        # Split
        train_texts = texts[num_val:]
        train_labels = labels[num_val:]
        val_texts = texts[:num_val]
        val_labels = labels[:num_val]
        
        # Tokenize
        train_encodings = tokenizer(
            list(train_texts),
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        )
        
        val_encodings = tokenizer(
            list(val_texts),
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        )
        
        # Create datasets
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_labels)
        )
        
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(val_labels)
        )
        
        # Create dataloaders
        def collate_fn(batch):
            input_ids, attention_mask, labels = zip(*batch)
            return {
                'input_ids': torch.stack(input_ids),
                'attention_mask': torch.stack(attention_mask),
                'labels': torch.stack(labels)
            }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Learn task
        logger.info(f"Learning task '{task_name}' from {len(train_texts)} texts...")
        
        self.model.learn_task(
            task_id=task_name,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            num_epochs=epochs or self.config.num_epochs,
            **kwargs
        )
        
        # Evaluate and record metrics
        final_metrics = self.model.evaluate_task(task_name, val_loader)
        
        self.metrics.record_task_performance(
            current_task=task_name,
            evaluated_task=task_name,
            accuracy=final_metrics["accuracy"],
            loss=final_metrics["loss"]
        )
        
        if task_name not in self.learned_tasks:
            self.learned_tasks.append(task_name)
        
        logger.info(f"Completed learning '{task_name}' - Accuracy: {final_metrics['accuracy']:.4f}")
        
        return final_metrics
    
    def predict(
        self,
        texts: Union[str, List[str]],
        task_name: str,
        return_probabilities: bool = False
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Make predictions for given texts.
        
        Args:
            texts: Input text(s) to classify
            task_name: Name of the task to use for prediction
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions (and probabilities if requested)
        """
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Make predictions
        results = self.model.predict(texts, task_name)
        
        if return_probabilities:
            return {
                "predictions": results["predictions"],
                "probabilities": results["probabilities"],
                "task_routing_confidence": results["task_routing_probs"]
            }
        else:
            return results["predictions"]
    
    def evaluate_all_tasks(self, test_data: Dict[str, Tuple[List[str], List[int]]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all learned tasks to check for catastrophic forgetting.
        
        Args:
            test_data: Dictionary mapping task names to (texts, labels) tuples
            
        Returns:
            Dictionary of evaluation metrics for each task
        """
        
        results = {}
        
        for task_name, (texts, labels) in test_data.items():
            if task_name not in self.learned_tasks:
                logger.warning(f"Task '{task_name}' not learned, skipping evaluation")
                continue
            
            # Prepare data
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # Secure default padding
            
            encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt"
            )
            
            # Create dataset
            from torch.utils.data import TensorDataset, DataLoader
            
            dataset = TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask'],
                torch.tensor(labels)
            )
            
            def collate_fn(batch):
                input_ids, attention_mask, labels = zip(*batch)
                return {
                    'input_ids': torch.stack(input_ids),
                    'attention_mask': torch.stack(attention_mask),
                    'labels': torch.stack(labels)
                }
            
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
            
            # Evaluate
            metrics = self.model.evaluate_task(task_name, dataloader)
            results[task_name] = metrics
            
            logger.info(f"Task '{task_name}' - Accuracy: {metrics['accuracy']:.4f}")
        
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        
        base_stats = self.model.get_memory_usage()
        
        return {
            "total_parameters": base_stats["total_parameters"],
            "frozen_parameters": base_stats["frozen_parameters"],
            "trainable_parameters": base_stats["trainable_parameters"],
            "num_learned_tasks": len(self.learned_tasks),
            "parameters_per_task": base_stats["avg_params_per_task"],
            "learned_tasks": self.learned_tasks.copy()
        }
    
    def save(self, path: str) -> None:
        """Save the model and configuration."""
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(save_path))
        
        # Save learned tasks info
        info = {
            "learned_tasks": self.learned_tasks,
            "memory_stats": self.get_memory_stats(),
            "metrics": self.metrics.get_summary() if hasattr(self.metrics, 'get_summary') else {}
        }
        
        with open(save_path / "model_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'SimpleContinualTransformer':
        """Load a saved model."""
        
        load_path = Path(path)
        
        # Load model info
        with open(load_path / "model_info.json", "r") as f:
            info = json.load(f)
        
        # Load config
        config = ContinualConfig.from_yaml(str(load_path / "config.yaml"))
        
        # Create instance
        instance = cls.__new__(cls)
        instance.config = config
        instance.model = ContinualTransformer.load_model(str(load_path), config)
        instance.metrics = ContinualMetrics()
        instance.learned_tasks = info["learned_tasks"]
        
        logger.info(f"Model loaded from {load_path}")
        
        return instance
    
    def optimize_for_inference(self, level: str = "balanced") -> Dict[str, Any]:
        """Optimize model for inference."""
        return self.model.optimize_for_inference(level)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information."""
        return self.model.get_system_status()


class QuickDemo:
    """Quick demonstration class for testing and examples."""
    
    @staticmethod
    def run_sentiment_example() -> Dict[str, Any]:
        """Run a quick sentiment analysis example."""
        
        # Initialize model
        model = SimpleContinualTransformer(device="cpu")  # Use CPU for demo
        
        # Add sentiment task
        model.add_task("sentiment", num_classes=2)
        
        # Sample data
        texts = [
            "I love this product! It's amazing!",
            "This is the worst thing I've ever bought.",
            "Great quality and fast shipping.",
            "Terrible customer service.",
            "Best purchase ever!",
            "Complete waste of money."
        ]
        labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
        
        # Learn task
        metrics = model.learn_from_texts("sentiment", texts, labels, epochs=2)
        
        # Test predictions
        test_texts = ["This is fantastic!", "I hate it."]
        predictions = model.predict(test_texts, "sentiment", return_probabilities=True)
        
        return {
            "training_metrics": metrics,
            "predictions": predictions,
            "memory_stats": model.get_memory_stats()
        }
    
    @staticmethod
    def run_continual_learning_example() -> Dict[str, Any]:
        """Run a continual learning example with multiple tasks."""
        
        # Initialize model
        model = SimpleContinualTransformer(device="cpu")
        
        # Define tasks
        tasks = [
            ("sentiment", ["Great product!", "Terrible quality", "Love it!", "Hate this"], [1, 0, 1, 0]),
            ("urgency", ["URGENT: Please respond", "When you have time", "Emergency!", "No rush"], [1, 0, 1, 0]),
            ("topic", ["Sports news today", "Stock market update", "Football game", "Economic report"], [0, 1, 0, 1])
        ]
        
        results = {}
        
        # Learn tasks sequentially
        for task_name, texts, labels in tasks:
            num_classes = len(set(labels))
            model.add_task(task_name, num_classes)
            
            metrics = model.learn_from_texts(task_name, texts, labels, epochs=2)
            results[f"{task_name}_metrics"] = metrics
        
        # Test all tasks
        test_data = {
            "sentiment": (["Amazing product!", "Worst purchase ever"], [1, 0]),
            "urgency": (["Please hurry!", "Take your time"], [1, 0]),
            "topic": (["Basketball game tonight", "Market trends today"], [0, 1])
        }
        
        evaluation_results = model.evaluate_all_tasks(test_data)
        results["final_evaluation"] = evaluation_results
        results["memory_stats"] = model.get_memory_stats()
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run quick examples
    print("Running sentiment example...")
    sentiment_results = QuickDemo.run_sentiment_example()
    print(f"Sentiment accuracy: {sentiment_results['training_metrics']['accuracy']:.4f}")
    
    print("\nRunning continual learning example...")
    continual_results = QuickDemo.run_continual_learning_example()
    print(f"Learned {continual_results['memory_stats']['num_learned_tasks']} tasks")
    print(f"Total parameters: {continual_results['memory_stats']['total_parameters']:,}")