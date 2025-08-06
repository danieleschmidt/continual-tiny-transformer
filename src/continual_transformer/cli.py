#!/usr/bin/env python3
"""Command Line Interface for Continual Tiny Transformer."""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader

from continual_transformer import ContinualTransformer, ContinualConfig
from continual_transformer.data.loaders import create_dataloader
from continual_transformer.tasks.manager import TaskType


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Continual Tiny Transformer - Zero-parameter continual learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on a new task
  continual-transformer train --task-id sentiment --data-path data/sentiment.json --task-type classification

  # Evaluate model performance
  continual-transformer evaluate --model-path ./outputs/model.pt --data-path data/test.json

  # Make predictions
  continual-transformer predict --model-path ./outputs/model.pt --text "This is great!" --task-id sentiment

  # Show model statistics
  continual-transformer info --model-path ./outputs/model.pt
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'], 
                       default='auto', help='Device to use')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--output-dir', type=str, default='./outputs', 
                       help='Output directory for models and logs')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model on a new task')
    train_parser.add_argument('--task-id', type=str, required=True,
                             help='Unique identifier for the task')
    train_parser.add_argument('--task-type', type=str, required=True,
                             choices=['classification', 'regression', 'sequence_labeling'],
                             help='Type of task')
    train_parser.add_argument('--data-path', type=str, required=True,
                             help='Path to training data (JSON format)')
    train_parser.add_argument('--eval-data-path', type=str,
                             help='Path to evaluation data')
    train_parser.add_argument('--model-path', type=str,
                             help='Path to existing model (for continual learning)')
    train_parser.add_argument('--num-labels', type=int, default=2,
                             help='Number of labels for classification tasks')
    train_parser.add_argument('--epochs', type=int, default=3,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5,
                             help='Learning rate')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--data-path', type=str, required=True,
                            help='Path to evaluation data')
    eval_parser.add_argument('--task-id', type=str,
                            help='Task ID to evaluate (if not provided, evaluates all tasks)')
    eval_parser.add_argument('--batch-size', type=int, default=16,
                            help='Evaluation batch size')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on text')
    predict_parser.add_argument('--model-path', type=str, required=True,
                               help='Path to trained model')
    predict_parser.add_argument('--task-id', type=str, required=True,
                               help='Task ID for prediction')
    predict_parser.add_argument('--text', type=str,
                               help='Text to predict on (use --file for file input)')
    predict_parser.add_argument('--file', type=str,
                               help='File containing text to predict on (one per line)')
    predict_parser.add_argument('--output', type=str,
                               help='Output file for predictions (JSON format)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model')
    info_parser.add_argument('--detailed', action='store_true',
                            help='Show detailed task information')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize new model')
    init_parser.add_argument('--model-name', type=str, default='distilbert-base-uncased',
                            help='Base transformer model name')
    init_parser.add_argument('--max-tasks', type=int, default=50,
                            help='Maximum number of tasks')
    
    return parser


def load_config(config_path: Optional[str], args: argparse.Namespace) -> ContinualConfig:
    """Load configuration from file or create from arguments."""
    if config_path and Path(config_path).exists():
        config = ContinualConfig.from_yaml(config_path)
    else:
        config = ContinualConfig()
    
    # Override with command line arguments
    if hasattr(args, 'device') and args.device != 'auto':
        config.device = args.device
    
    if hasattr(args, 'output_dir'):
        config.output_dir = args.output_dir
    
    if hasattr(args, 'batch_size'):
        config.batch_size = args.batch_size
    
    if hasattr(args, 'learning_rate'):
        config.learning_rate = args.learning_rate
    
    if hasattr(args, 'epochs'):
        config.num_epochs = args.epochs
    
    return config


def train_command(args: argparse.Namespace, config: ContinualConfig) -> int:
    """Execute train command."""
    try:
        # Load or create model
        if args.model_path and Path(args.model_path).exists():
            print(f"Loading existing model from {args.model_path}")
            model = ContinualTransformer.load_model(args.model_path, config)
        else:
            print("Creating new model")
            model = ContinualTransformer(config)
        
        # Register task
        print(f"Registering task '{args.task_id}' (type: {args.task_type})")
        model.register_task(
            task_id=args.task_id,
            num_labels=args.num_labels,
            task_type=args.task_type
        )
        
        # Load training data
        print(f"Loading training data from {args.data_path}")
        train_dataloader = create_dataloader(
            data_path=args.data_path,
            batch_size=config.batch_size,
            tokenizer_name=config.model_name,
            max_length=config.max_sequence_length,
            split='train'
        )
        
        # Load evaluation data if provided
        eval_dataloader = None
        if args.eval_data_path:
            print(f"Loading evaluation data from {args.eval_data_path}")
            eval_dataloader = create_dataloader(
                data_path=args.eval_data_path,
                batch_size=config.batch_size,
                tokenizer_name=config.model_name,
                max_length=config.max_sequence_length,
                split='eval'
            )
        
        # Train model
        print(f"Training model on task '{args.task_id}'...")
        model.learn_task(
            task_id=args.task_id,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            num_epochs=config.num_epochs
        )
        
        # Save model
        save_path = Path(config.output_dir) / "model.pt"
        print(f"Saving model to {save_path}")
        model.save_model(str(save_path.parent))
        
        # Show memory usage
        memory_stats = model.get_memory_usage()
        print(f"\nModel Statistics:")
        print(f"Total parameters: {memory_stats['total_parameters']:,}")
        print(f"Trainable parameters: {memory_stats['trainable_parameters']:,}")
        print(f"Number of tasks: {memory_stats['num_tasks']}")
        print(f"Average parameters per task: {memory_stats['avg_params_per_task']:,}")
        
        print(f"\nTask '{args.task_id}' training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}")
        logging.exception("Training failed")
        return 1


def evaluate_command(args: argparse.Namespace, config: ContinualConfig) -> int:
    """Execute evaluate command."""
    try:
        # Load model
        print(f"Loading model from {args.model_path}")
        model = ContinualTransformer.load_model(args.model_path, config)
        
        # Load evaluation data
        print(f"Loading evaluation data from {args.data_path}")
        eval_dataloader = create_dataloader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            tokenizer_name=config.model_name,
            max_length=config.max_sequence_length,
            split='eval'
        )
        
        # Evaluate specific task or all tasks
        if args.task_id:
            print(f"Evaluating task '{args.task_id}'")
            metrics = model.evaluate_task(args.task_id, eval_dataloader)
            
            print(f"\nTask '{args.task_id}' Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Loss: {metrics['loss']:.4f}")
            print(f"Samples: {metrics['total_samples']}")
            
        else:
            print("Evaluating all tasks")
            all_tasks = list(model.adapters.keys())
            
            for task_id in all_tasks:
                model.set_current_task(task_id)
                metrics = model.evaluate_task(task_id, eval_dataloader)
                print(f"Task '{task_id}': Accuracy={metrics['accuracy']:.4f}, Loss={metrics['loss']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        logging.exception("Evaluation failed")
        return 1


def predict_command(args: argparse.Namespace, config: ContinualConfig) -> int:
    """Execute predict command."""
    try:
        # Load model
        print(f"Loading model from {args.model_path}")
        model = ContinualTransformer.load_model(args.model_path, config)
        
        # Get input texts
        texts = []
        if args.text:
            texts = [args.text]
        elif args.file:
            with open(args.file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            # Interactive mode
            print("Enter text for prediction (Ctrl+C to exit):")
            while True:
                try:
                    text = input("> ")
                    if text.strip():
                        texts = [text.strip()]
                        break
                except KeyboardInterrupt:
                    print("\nExiting...")
                    return 0
        
        if not texts:
            print("No input text provided")
            return 1
        
        # Make predictions
        results = []
        for text in texts:
            print(f"Predicting: {text[:100]}...")
            prediction = model.predict(text, args.task_id)
            results.append({
                "text": text,
                "prediction": prediction["predictions"][0],
                "probabilities": prediction["probabilities"][0],
                "task_id": args.task_id
            })
            
            print(f"Prediction: {prediction['predictions'][0]}")
            print(f"Confidence: {max(prediction['probabilities'][0]):.4f}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        logging.exception("Prediction failed")
        return 1


def info_command(args: argparse.Namespace, config: ContinualConfig) -> int:
    """Execute info command."""
    try:
        # Load model
        print(f"Loading model from {args.model_path}")
        model = ContinualTransformer.load_model(args.model_path, config)
        
        # Show basic model info
        memory_stats = model.get_memory_usage()
        print(f"\nModel Information:")
        print(f"==================")
        print(f"Base model: {config.model_name}")
        print(f"Device: {config.device}")
        print(f"Total parameters: {memory_stats['total_parameters']:,}")
        print(f"Trainable parameters: {memory_stats['trainable_parameters']:,}")
        print(f"Frozen parameters: {memory_stats['frozen_parameters']:,}")
        print(f"Parameter efficiency: {memory_stats['trainable_parameters']/memory_stats['total_parameters']:.1%}")
        
        # Task information
        print(f"\nTask Information:")
        print(f"=================")
        print(f"Number of tasks: {memory_stats['num_tasks']}")
        print(f"Max tasks: {config.max_tasks}")
        print(f"Average parameters per task: {memory_stats['avg_params_per_task']:,}")
        
        # List all tasks
        if model.adapters:
            print(f"\nRegistered Tasks:")
            for task_id in model.adapters.keys():
                task = model.task_manager.get_task(task_id)
                if task:
                    print(f"  - {task_id}: {task.task_type.value} ({task.num_labels} labels) - {task.status.value}")
        
        # Show detailed info if requested
        if args.detailed and hasattr(model, 'task_manager'):
            print(f"\nTask Statistics:")
            print(f"================")
            summary = model.task_manager.get_task_summary()
            print(summary)
        
        return 0
        
    except Exception as e:
        print(f"Error loading model info: {e}")
        logging.exception("Info command failed")
        return 1


def init_command(args: argparse.Namespace, config: ContinualConfig) -> int:
    """Execute init command."""
    try:
        # Update config with arguments
        config.model_name = args.model_name
        config.max_tasks = args.max_tasks
        
        # Create model
        print(f"Initializing new model with {args.model_name}")
        model = ContinualTransformer(config)
        
        # Save model and config
        save_path = Path(config.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model.save_model(str(save_path))
        print(f"Model initialized and saved to {save_path}")
        
        # Show model info
        memory_stats = model.get_memory_usage()
        print(f"\nInitialized Model:")
        print(f"Base model: {config.model_name}")
        print(f"Max tasks: {config.max_tasks}")
        print(f"Total parameters: {memory_stats['total_parameters']:,}")
        
        return 0
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        logging.exception("Init command failed")
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Load configuration
        config = load_config(args.config, args)
        
        # Execute command
        if args.command == 'train':
            return train_command(args, config)
        elif args.command == 'evaluate':
            return evaluate_command(args, config)
        elif args.command == 'predict':
            return predict_command(args, config)
        elif args.command == 'info':
            return info_command(args, config)
        elif args.command == 'init':
            return init_command(args, config)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.exception("CLI execution failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())