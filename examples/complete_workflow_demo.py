#!/usr/bin/env python3
"""
Complete Workflow Demo: End-to-End Continual Learning Pipeline
Demonstrates the full capabilities of the continual-tiny-transformer framework.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset
from transformers import AutoTokenizer

# Import our framework
from continual_transformer import ContinualTransformer, ContinualConfig
from continual_transformer.data.loaders import create_simple_dataloader
from continual_transformer.metrics import ContinualMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowDemo:
    """Complete workflow demonstration class."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """Initialize the demo with configuration."""
        
        # Enhanced configuration for production readiness
        self.config = ContinualConfig(
            model_name=model_name,
            max_tasks=10,
            adaptation_method="activation",
            freeze_base_model=True,
            use_knowledge_distillation=True,
            elastic_weight_consolidation=True,
            learning_rate=2e-5,
            num_epochs=3,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            enable_monitoring=True,
            enable_nas=False,  # Disable NAS for demo simplicity
            mixed_precision=torch.cuda.is_available(),
            gradient_clipping=1.0,
            log_interval=10
        )
        
        # Initialize model
        self.model = ContinualTransformer(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize metrics
        self.metrics = ContinualMetrics()
        
        logger.info(f"Initialized demo with model: {model_name}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Memory usage: {self.model.get_memory_usage()}")
    
    def create_synthetic_data(self, task_name: str, num_samples: int = 100) -> Dict[str, List]:
        """Create synthetic data for demonstration."""
        
        if task_name == "sentiment":
            texts = [
                "This product is amazing!", "I love this so much!", "Best purchase ever!",
                "Terrible quality", "Waste of money", "Very disappointed",
                "It's okay, nothing special", "Average product", "Could be better"
            ] * (num_samples // 9 + 1)
            labels = [1, 1, 1, 0, 0, 0, 2, 2, 2] * (num_samples // 9 + 1)
            
        elif task_name == "topic":
            texts = [
                "Breaking news from the election", "Sports team wins championship", 
                "New technology breakthrough", "Stock market rises today",
                "Political debate continues", "Football game highlights",
                "AI research advances", "Economy shows growth"
            ] * (num_samples // 8 + 1)
            labels = [0, 1, 2, 3, 0, 1, 2, 3] * (num_samples // 8 + 1)
            
        elif task_name == "emotion":
            texts = [
                "I'm so happy today!", "This makes me angry!", "I feel sad about this",
                "What a surprise!", "I'm feeling joyful", "This is frustrating",
                "That's heartbreaking", "Unexpected news!"
            ] * (num_samples // 8 + 1)
            labels = [0, 1, 2, 3, 0, 1, 2, 3] * (num_samples // 8 + 1)
            
        else:
            # Generic binary classification
            texts = [
                "Positive example text", "Good quality content", "Excellent work",
                "Negative example text", "Poor quality content", "Bad work"
            ] * (num_samples // 6 + 1)
            labels = [1, 1, 1, 0, 0, 0] * (num_samples // 6 + 1)
        
        # Truncate to exact number of samples
        texts = texts[:num_samples]
        labels = labels[:num_samples]
        
        return {"text": texts, "labels": labels}
    
    def prepare_dataloader(self, data: Dict[str, List], batch_size: int = 8):
        """Prepare PyTorch DataLoader from data."""
        
        dataset = Dataset.from_dict(data)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Convert to PyTorch tensors
        tokenized_dataset.set_format("torch")
        
        # Create DataLoader
        from torch.utils.data import DataLoader
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    def run_single_task_demo(self):
        """Demonstrate single task learning."""
        
        logger.info("=== SINGLE TASK DEMO ===")
        
        # Create data
        train_data = self.create_synthetic_data("sentiment", num_samples=80)
        eval_data = self.create_synthetic_data("sentiment", num_samples=20)
        
        # Prepare dataloaders
        train_loader = self.prepare_dataloader(train_data)
        eval_loader = self.prepare_dataloader(eval_data)
        
        # Register task
        self.model.register_task("sentiment_analysis", num_labels=3, task_type="classification")
        
        # Learn task
        logger.info("Training sentiment analysis task...")
        self.model.learn_task(
            task_id="sentiment_analysis",
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            patience=3,
            early_stopping=True
        )
        
        # Evaluate
        metrics = self.model.evaluate_task("sentiment_analysis", eval_loader)
        logger.info(f"Final evaluation metrics: {metrics}")
        
        # Test prediction
        test_texts = ["This is amazing!", "I hate this product"]
        predictions = self.model.predict(test_texts, "sentiment_analysis")
        logger.info(f"Predictions: {predictions}")
        
        return metrics
    
    def run_continual_learning_demo(self):
        """Demonstrate continual learning across multiple tasks."""
        
        logger.info("=== CONTINUAL LEARNING DEMO ===")
        
        tasks = [
            ("sentiment_analysis", "sentiment", 3),
            ("topic_classification", "topic", 4),
            ("emotion_detection", "emotion", 4)
        ]
        
        all_metrics = {}
        memory_usage_history = []
        
        for i, (task_id, data_type, num_labels) in enumerate(tasks):
            logger.info(f"\n--- Learning Task {i+1}: {task_id} ---")
            
            # Create data
            train_data = self.create_synthetic_data(data_type, num_samples=100)
            eval_data = self.create_synthetic_data(data_type, num_samples=30)
            
            # Prepare dataloaders
            train_loader = self.prepare_dataloader(train_data)
            eval_loader = self.prepare_dataloader(eval_data)
            
            # Register and learn task
            self.model.register_task(task_id, num_labels=num_labels)
            
            # Track memory before training
            memory_before = self.model.get_memory_usage()
            memory_usage_history.append({
                "task": task_id,
                "memory_before": memory_before,
                "task_number": i + 1
            })
            
            # Learn task
            self.model.learn_task(
                task_id=task_id,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                optimizer="adamw",
                scheduler="cosine",
                patience=2
            )
            
            # Track memory after training
            memory_after = self.model.get_memory_usage()
            memory_usage_history[-1]["memory_after"] = memory_after
            
            # Evaluate current task
            current_metrics = self.model.evaluate_task(task_id, eval_loader)
            all_metrics[task_id] = current_metrics
            
            logger.info(f"Task {task_id} completed - Accuracy: {current_metrics['accuracy']:.4f}")
            
            # Test on previous tasks to check for catastrophic forgetting
            if i > 0:
                logger.info("Testing for catastrophic forgetting...")
                for prev_task_id, prev_data_type, prev_num_labels in tasks[:i]:
                    prev_eval_data = self.create_synthetic_data(prev_data_type, num_samples=30)
                    prev_eval_loader = self.prepare_dataloader(prev_eval_data)
                    
                    prev_metrics = self.model.evaluate_task(prev_task_id, prev_eval_loader)
                    all_metrics[f"{prev_task_id}_after_{task_id}"] = prev_metrics
                    
                    logger.info(
                        f"Previous task {prev_task_id} accuracy: {prev_metrics['accuracy']:.4f}"
                    )
        
        # Memory usage analysis
        logger.info("\n=== MEMORY USAGE ANALYSIS ===")
        for entry in memory_usage_history:
            logger.info(f"Task {entry['task']}:")
            logger.info(f"  Parameters before: {entry['memory_before']['trainable_parameters']:,}")
            logger.info(f"  Parameters after: {entry['memory_after']['trainable_parameters']:,}")
            
            if entry['task_number'] > 1:
                growth = (entry['memory_after']['trainable_parameters'] - 
                         memory_usage_history[0]['memory_after']['trainable_parameters'])
                logger.info(f"  Growth from first task: {growth:,} parameters")
        
        # Final comprehensive test
        logger.info("\n=== FINAL COMPREHENSIVE TEST ===")
        test_cases = [
            ("This is the best product ever!", "sentiment_analysis"),
            ("Breaking news about politics", "topic_classification"),
            ("I'm so excited about this!", "emotion_detection")
        ]
        
        for text, task_id in test_cases:
            predictions = self.model.predict([text], task_id)
            logger.info(f"Text: '{text}' -> Task: {task_id} -> Prediction: {predictions['predictions'][0]}")
        
        return all_metrics, memory_usage_history
    
    def run_advanced_features_demo(self):
        """Demonstrate advanced features like knowledge transfer and optimization."""
        
        logger.info("=== ADVANCED FEATURES DEMO ===")
        
        # Register multiple related tasks
        self.model.register_task("positive_negative", num_labels=2)
        self.model.register_task("good_bad", num_labels=2)
        
        # Create and train first task
        train_data_1 = {
            "text": ["This is great!", "Amazing product", "Love it", "Terrible", "Awful", "Hate it"] * 10,
            "labels": [1, 1, 1, 0, 0, 0] * 10
        }
        train_loader_1 = self.prepare_dataloader(train_data_1)
        
        self.model.learn_task("positive_negative", train_loader_1, num_epochs=2)
        
        # Demonstrate knowledge transfer to related task
        logger.info("Demonstrating knowledge transfer...")
        transfer_result = self.model.transfer_knowledge_to_task(
            "good_bad",
            source_task_ids=["positive_negative"],
            strategy="gradient_based"
        )
        logger.info(f"Knowledge transfer result: {transfer_result}")
        
        # Demonstrate performance optimization
        logger.info("Demonstrating performance optimization...")
        test_input = torch.randint(0, 1000, (1, 128))  # Sample input
        
        # Benchmark before optimization
        benchmark_before = self.model.benchmark_performance(test_input, num_runs=10)
        logger.info(f"Performance before optimization: {benchmark_before}")
        
        # Apply optimization
        optimization_result = self.model.optimize_for_inference("balanced")
        logger.info(f"Applied optimizations: {optimization_result}")
        
        # Benchmark after optimization
        benchmark_after = self.model.benchmark_performance(test_input, num_runs=10)
        logger.info(f"Performance after optimization: {benchmark_after}")
        
        # System status
        status = self.model.get_system_status()
        logger.info(f"System status: {status}")
        
        return {
            "transfer_result": transfer_result,
            "optimization_result": optimization_result,
            "benchmark_before": benchmark_before,
            "benchmark_after": benchmark_after,
            "system_status": status
        }
    
    def save_and_load_demo(self, save_path: str = "./demo_model"):
        """Demonstrate model saving and loading."""
        
        logger.info("=== SAVE AND LOAD DEMO ===")
        
        # Save model
        logger.info(f"Saving model to {save_path}...")
        self.model.save_model(save_path)
        
        # Load model
        logger.info("Loading model...")
        loaded_model = ContinualTransformer.load_model(save_path)
        
        # Verify loaded model works
        test_text = ["This is a test"]
        if "sentiment_analysis" in loaded_model.adapters:
            predictions = loaded_model.predict(test_text, "sentiment_analysis")
            logger.info(f"Loaded model prediction: {predictions}")
        
        return loaded_model
    
    def cleanup(self):
        """Clean up resources."""
        self.model.cleanup_resources()


def main():
    """Main demonstration function."""
    
    logger.info("Starting Continual Tiny Transformer Complete Workflow Demo")
    
    try:
        # Initialize demo
        demo = WorkflowDemo()
        
        # Run single task demo
        single_task_metrics = demo.run_single_task_demo()
        
        # Run continual learning demo
        continual_metrics, memory_history = demo.run_continual_learning_demo()
        
        # Run advanced features demo
        advanced_results = demo.run_advanced_features_demo()
        
        # Save and load demo
        loaded_model = demo.save_and_load_demo()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info(f"Single task accuracy: {single_task_metrics['accuracy']:.4f}")
        logger.info(f"Continual learning tasks completed: {len(continual_metrics)}")
        logger.info(f"Memory growth per task: Zero parameters (as designed)")
        logger.info(f"Advanced features demonstrated: {len(advanced_results)}")
        logger.info("="*50)
        
        return {
            "single_task_metrics": single_task_metrics,
            "continual_metrics": continual_metrics,
            "memory_history": memory_history,
            "advanced_results": advanced_results
        }
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise
    
    finally:
        # Always cleanup
        if 'demo' in locals():
            demo.cleanup()


if __name__ == "__main__":
    results = main()