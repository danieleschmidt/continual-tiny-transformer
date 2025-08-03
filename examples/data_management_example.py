#!/usr/bin/env python3
"""
Data Management Example for Continual Tiny Transformer.

This example demonstrates:
1. Advanced data loading and processing
2. Storage management for tasks and models
3. Memory replay mechanisms
4. Continual batch sampling strategies
5. Comprehensive metrics tracking
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from continual_transformer import ContinualTransformer, ContinualConfig
from continual_transformer.data import (
    ContinualDataLoader,
    TaskDataset,
    MemoryReplayDataLoader,
    TextProcessor,
    TaskBatchProcessor, 
    ContinualBatchSampler,
    TaskDataStorage,
    ModelCheckpointManager,
    MetricsStorage,
    create_synthetic_task_data
)
from continual_transformer.metrics import ContinualMetrics
from transformers import AutoTokenizer


def setup_storage_systems(base_dir: str = "./data_example"):
    """Set up storage systems for data, models, and metrics."""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Initialize storage systems
    task_storage = TaskDataStorage(base_path / "tasks")
    checkpoint_manager = ModelCheckpointManager(base_path / "checkpoints", max_checkpoints=5)
    metrics_storage = MetricsStorage(base_path / "metrics")
    
    print(f"📁 Storage systems initialized at {base_path}")
    return task_storage, checkpoint_manager, metrics_storage


def create_diverse_datasets():
    """Create diverse synthetic datasets for different domains."""
    datasets = {}
    
    # Sentiment Analysis Task
    datasets["sentiment"] = create_synthetic_task_data(
        task_id="sentiment",
        num_samples=500,
        num_classes=3,
        class_names=["negative", "neutral", "positive"]
    )
    
    # Topic Classification Task  
    datasets["topic"] = create_synthetic_task_data(
        task_id="topic",
        num_samples=600,
        num_classes=4,
        class_names=["technology", "sports", "politics", "entertainment"]
    )
    
    # Intent Detection Task
    datasets["intent"] = create_synthetic_task_data(
        task_id="intent",
        num_samples=400,
        num_classes=5,
        class_names=["book_flight", "cancel_booking", "check_status", "get_help", "other"]
    )
    
    # Language Detection Task
    datasets["language"] = create_synthetic_task_data(
        task_id="language",
        num_samples=700,
        num_classes=3,
        class_names=["english", "spanish", "french"]
    )
    
    print(f"📊 Created {len(datasets)} diverse datasets")
    return datasets


def demonstrate_advanced_data_processing():
    """Demonstrate advanced data processing capabilities."""
    print("\n" + "="*60)
    print("🔄 ADVANCED DATA PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Initialize text processor
    tokenizer_name = "distilbert-base-uncased"
    text_processor = TextProcessor(
        tokenizer_name=tokenizer_name,
        max_length=128
    )
    
    # Initialize task batch processor
    batch_processor = TaskBatchProcessor(text_processor)
    
    # Create sample datasets
    datasets = create_diverse_datasets()
    
    # Register task labels
    for task_id, data in datasets.items():
        unique_labels = list(set(item['label_name'] for item in data))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        batch_processor.register_task_labels(task_id, label_mapping)
        print(f"  📝 Registered {task_id}: {label_mapping}")
    
    # Process sample batches
    print(f"\n🔄 Processing sample batches...")
    
    for task_id, data in datasets.items():
        # Sample a batch
        batch_data = np.random.choice(data, size=8, replace=False).tolist()
        
        # Process batch
        processed_batch = batch_processor.process_task_batch(batch_data, task_id)
        
        print(f"  ✅ {task_id}: {processed_batch['batch_size']} samples processed")
        print(f"    - Input shape: {processed_batch['input_ids'].shape}")
        print(f"    - Labels shape: {processed_batch['labels'].shape}")
    
    # Demonstrate multi-task batch processing
    print(f"\n🎯 Multi-task batch processing...")
    
    # Create mixed batch
    mixed_batch = []
    for task_id, data in datasets.items():
        # Add 2 samples from each task
        task_samples = np.random.choice(data, size=2, replace=False).tolist()
        mixed_batch.extend(task_samples)
    
    # Process mixed batch
    multi_task_batch = batch_processor.process_multi_task_batch(mixed_batch)
    print(f"  ✅ Multi-task batch: {multi_task_batch['batch_size']} samples, {multi_task_batch['num_tasks']} tasks")
    
    # Show statistics
    stats = batch_processor.get_task_statistics()
    print(f"\n📈 Processing Statistics:")
    for task_id, task_stats in stats.items():
        print(f"  {task_id}:")
        print(f"    - Batches processed: {task_stats['processing_stats']['num_batches']}")
        print(f"    - Total samples: {task_stats['processing_stats']['total_samples']}")
        print(f"    - Unique tokens: {task_stats['text_stats']['unique_tokens']}")
        print(f"    - Avg text length: {task_stats['text_stats']['avg_length']:.1f}")
    
    return text_processor, batch_processor, datasets


def demonstrate_storage_management(datasets):
    """Demonstrate storage management capabilities."""
    print("\n" + "="*60)
    print("💾 STORAGE MANAGEMENT DEMONSTRATION") 
    print("="*60)
    
    # Set up storage systems
    task_storage, checkpoint_manager, metrics_storage = setup_storage_systems()
    
    # Store datasets
    print(f"\n📥 Storing datasets...")
    for task_id, data in datasets.items():
        metadata = {
            'task_type': 'classification',
            'description': f'Synthetic {task_id} classification data',
            'num_classes': len(set(item['label'] for item in data)),
            'created_by': 'data_management_example.py'
        }
        
        task_dir = task_storage.save_task_data(task_id, data, metadata)
        print(f"  ✅ Stored {task_id}: {len(data)} samples at {task_dir}")
    
    # List stored tasks
    stored_tasks = task_storage.list_tasks()
    print(f"\n📋 Stored tasks: {len(stored_tasks)}")
    for task_info in stored_tasks:
        print(f"  - {task_info['task_id']}: {task_info['num_samples']} samples, {task_info['num_classes']} classes")
    
    # Demonstrate loading
    print(f"\n📤 Loading sample task...")
    sample_task = "sentiment"
    loaded_data, loaded_metadata = task_storage.load_task_data(sample_task)
    print(f"  ✅ Loaded {sample_task}: {len(loaded_data)} samples")
    print(f"    Metadata: {loaded_metadata['description']}")
    
    return task_storage, checkpoint_manager, metrics_storage


def demonstrate_continual_data_loading(datasets, text_processor):
    """Demonstrate continual data loading with replay."""
    print("\n" + "="*60)
    print("🔄 CONTINUAL DATA LOADING DEMONSTRATION")
    print("="*60)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Initialize continual data loader
    continual_loader = ContinualDataLoader(
        batch_size=16,
        num_workers=0,  # Single-threaded for demo
        pin_memory=False
    )
    
    # Initialize memory replay loader
    replay_loader = MemoryReplayDataLoader(
        buffer_size=100,
        replay_ratio=0.3,
        batch_size=16,
        strategy="random"
    )
    
    print(f"\n📚 Adding task datasets...")
    
    # Add datasets to continual loader
    task_loaders = {}
    for task_id, data in datasets.items():
        # Create TaskDataset
        dataset = TaskDataset(
            data=data,
            task_id=task_id,
            tokenizer=tokenizer,
            max_length=128
        )
        
        # Add to continual loader
        train_loader, val_loader = continual_loader.add_task_dataset(
            task_id=task_id,
            dataset=dataset,
            train_ratio=0.8
        )
        
        task_loaders[task_id] = {
            'train': train_loader,
            'val': val_loader,
            'dataset': dataset
        }
        
        print(f"  ✅ Added {task_id}: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Show dataset statistics
        stats = dataset.get_statistics()
        print(f"    - Label distribution: {stats['label_distribution']}")
        print(f"    - Avg text length: {stats['avg_text_length']:.1f} words")
    
    # Demonstrate replay mechanism
    print(f"\n🔄 Memory replay demonstration...")
    
    for i, (task_id, task_info) in enumerate(task_loaders.items()):
        train_loader = task_info['train']
        
        print(f"\n  Learning task {i+1}: {task_id}")
        
        # Add some samples to replay buffer
        for batch_idx, batch in enumerate(train_loader):
            replay_loader.add_to_memory(task_id, batch)
            
            # Only add first few batches for demo
            if batch_idx >= 2:
                break
        
        print(f"    - Added samples to replay buffer")
        
        # Demonstrate replay sampling (exclude current task)
        if i > 0:  # Only after first task
            replay_batch = replay_loader.sample_replay_batch(
                exclude_task=task_id,
                num_samples=8
            )
            
            if replay_batch:
                print(f"    - Sampled replay batch: {replay_batch['input_ids'].shape[0]} samples")
                print(f"    - Tasks in replay: {set(replay_batch['task_ids'])}")
    
    # Show memory statistics
    memory_stats = replay_loader.get_memory_stats()
    print(f"\n📊 Memory replay statistics:")
    print(f"  - Buffer size: {memory_stats['buffer_size']}")
    print(f"  - Strategy: {memory_stats['strategy']}")
    print(f"  - Task buffers:")
    for task_id, buffer_stats in memory_stats['task_buffers'].items():
        print(f"    - {task_id}: {buffer_stats['samples']} samples ({buffer_stats['utilization']:.1%} full)")
    
    # Demonstrate multi-task loading
    print(f"\n🎯 Multi-task data loading...")
    multi_task_loader = continual_loader.get_multi_task_loader(
        task_ids=list(datasets.keys()),
        split="train",
        mixing_ratio=0.25
    )
    
    print(f"  ✅ Created multi-task loader with {len(multi_task_loader)} batches")
    
    # Sample from multi-task loader
    sample_batch = next(iter(multi_task_loader))
    task_ids_in_batch = sample_batch['task_ids']
    unique_tasks = set(task_ids_in_batch)
    
    print(f"  - Sample batch: {len(task_ids_in_batch)} samples from {len(unique_tasks)} tasks")
    print(f"  - Tasks in batch: {unique_tasks}")
    
    return continual_loader, replay_loader, task_loaders


def demonstrate_advanced_sampling():
    """Demonstrate advanced batch sampling strategies."""
    print("\n" + "="*60)
    print("🎲 ADVANCED SAMPLING STRATEGIES")
    print("="*60)
    
    # Create datasets for sampling
    datasets = create_diverse_datasets()
    
    # Initialize advanced batch sampler
    sampler = ContinualBatchSampler(
        datasets=datasets,
        batch_size=16,
        replay_ratio=0.3,
        curriculum_strategy="difficulty"
    )
    
    # Set task difficulties (simulate)
    difficulties = {
        "sentiment": 0.2,    # Easy
        "language": 0.4,     # Medium-Easy  
        "topic": 0.6,        # Medium-Hard
        "intent": 0.8        # Hard
    }
    
    sampler.set_task_difficulties(difficulties)
    print(f"📈 Set task difficulties: {difficulties}")
    
    # Get curriculum order
    curriculum_order = sampler.get_curriculum_order()
    print(f"🎯 Curriculum order: {curriculum_order}")
    
    # Demonstrate continual sampling
    print(f"\n🔄 Continual sampling demonstration...")
    
    for i, task_id in enumerate(curriculum_order):
        print(f"\n  Learning task {i+1}: {task_id}")
        
        # Update replay buffer
        sampler.update_replay_buffer(task_id, datasets[task_id], buffer_size=50)
        
        # Sample training batches
        for batch_num in range(3):  # Demo with 3 batches
            continual_batch = sampler.sample_continual_batch(
                current_task=task_id,
                current_data=datasets[task_id],
                include_replay=(i > 0)  # Include replay after first task
            )
            
            # Analyze batch composition
            total_samples = len(continual_batch)
            current_task_samples = sum(1 for item in continual_batch if item.get('task_id') == task_id)
            replay_samples = sum(1 for item in continual_batch if item.get('is_replay', False))
            
            print(f"    Batch {batch_num + 1}: {total_samples} samples")
            print(f"      - Current task ({task_id}): {current_task_samples}")
            print(f"      - Replay samples: {replay_samples}")
            
            if replay_samples > 0:
                replay_tasks = set(
                    item['task_id'] for item in continual_batch 
                    if item.get('is_replay', False)
                )
                print(f"      - Replay from: {replay_tasks}")
    
    # Show sampling statistics
    sampling_stats = sampler.get_sampling_statistics()
    print(f"\n📊 Sampling statistics:")
    print(f"  - Batch size: {sampling_stats['batch_size']}")
    print(f"  - Replay ratio: {sampling_stats['replay_ratio']}")
    print(f"  - Curriculum strategy: {sampling_stats['curriculum_strategy']}")
    print(f"  - Replay buffer sizes: {sampling_stats['replay_buffer_sizes']}")
    
    return sampler


def demonstrate_model_checkpointing():
    """Demonstrate model checkpointing and storage."""
    print("\n" + "="*60)
    print("💾 MODEL CHECKPOINTING DEMONSTRATION")
    print("="*60)
    
    # Initialize model and checkpoint manager
    config = ContinualConfig(
        model_name="distilbert-base-uncased",
        max_tasks=4,
        batch_size=8,
        num_epochs=1,  # Short for demo
        output_dir="./data_example/outputs"
    )
    
    model = ContinualTransformer(config)
    checkpoint_manager = ModelCheckpointManager("./data_example/checkpoints", max_checkpoints=3)
    
    # Simulate training and checkpointing
    print(f"\n🎯 Simulating training with checkpointing...")
    
    tasks = ["sentiment", "topic", "intent"] 
    
    for i, task_id in enumerate(tasks):
        print(f"\n  Training task {i+1}: {task_id}")
        
        # Register task (simulate different number of labels)
        num_labels = 3 + i  # Varying number of labels
        model.register_task(task_id, num_labels)
        
        # Simulate training epochs
        for epoch in range(2):
            # Simulate training metrics
            loss = 2.0 - (epoch * 0.5) - (i * 0.2)  # Decreasing loss
            accuracy = 0.3 + (epoch * 0.2) + (i * 0.1)  # Increasing accuracy
            step = epoch * 100 + i * 200
            
            print(f"    Epoch {epoch + 1}: loss={loss:.3f}, accuracy={accuracy:.3f}")
            
            # Save checkpoint
            checkpoint_id = checkpoint_manager.save_checkpoint(
                model=model,
                task_id=task_id,
                epoch=epoch,
                step=step,
                loss=loss,
                accuracy=accuracy,
                additional_data={
                    'learning_rate': config.learning_rate,
                    'task_order': tasks[:i+1]
                }
            )
            
            print(f"      ✅ Saved checkpoint: {checkpoint_id}")
    
    # Show checkpoint statistics
    print(f"\n📊 Checkpoint statistics:")
    storage_stats = checkpoint_manager.get_storage_stats()
    
    print(f"  - Total checkpoints: {storage_stats['total_checkpoints']}")
    print(f"  - Total size: {storage_stats['total_size_mb']:.1f} MB")
    print(f"  - Best checkpoint: {storage_stats['best_checkpoint']['checkpoint_id'] if storage_stats['best_checkpoint'] else 'None'}")
    
    print(f"\n📋 Checkpoints by task:")
    for task_id, task_stats in storage_stats['checkpoints_by_task'].items():
        print(f"  - {task_id}: {task_stats['count']} checkpoints, {task_stats['size_mb']:.1f} MB, best acc: {task_stats['best_accuracy']:.3f}")
    
    # Demonstrate loading
    print(f"\n📤 Loading best checkpoint...")
    best_checkpoint_data = checkpoint_manager.load_best_checkpoint()
    if best_checkpoint_data:
        print(f"  ✅ Loaded best checkpoint")
        print(f"    - Task: {best_checkpoint_data['task_id']}")
        print(f"    - Accuracy: {best_checkpoint_data['accuracy']:.3f}")
        print(f"    - Task order: {best_checkpoint_data.get('task_order', [])}")
    
    return checkpoint_manager


def demonstrate_metrics_tracking():
    """Demonstrate comprehensive metrics tracking."""
    print("\n" + "="*60)
    print("📊 METRICS TRACKING DEMONSTRATION")
    print("="*60)
    
    # Initialize metrics storage
    metrics_storage = MetricsStorage("./data_example/metrics")
    
    # Initialize continual metrics
    continual_metrics = ContinualMetrics()
    
    # Generate run ID
    run_id = f"demo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"🏃 Starting run: {run_id}")
    
    # Simulate continual learning with metrics
    tasks = ["sentiment", "topic", "intent", "language"]
    
    # Record baseline performance (optional)
    baseline_accuracies = {"sentiment": 0.85, "topic": 0.78, "intent": 0.82, "language": 0.90}
    for task_id, accuracy in baseline_accuracies.items():
        continual_metrics.record_baseline_performance(task_id, accuracy)
    
    print(f"\n📈 Simulating continual learning...")
    
    for i, current_task in enumerate(tasks):
        print(f"\n  Learning task {i+1}: {current_task}")
        
        # Add to task sequence
        continual_metrics.add_task_to_sequence(current_task)
        
        # Simulate parameter growth (minimal for our approach)
        base_params = 66_000_000  # DistilBERT base
        adapter_params = 50_000   # Small adapter per task
        total_params = base_params + (i + 1) * adapter_params
        continual_metrics.record_parameter_count(total_params)
        
        # Simulate training epochs
        for epoch in range(2):
            step = i * 200 + epoch * 100
            
            # Training metrics
            train_loss = 2.0 - (epoch * 0.3) - (i * 0.1)
            train_acc = 0.4 + (epoch * 0.2) + (i * 0.05)
            
            train_metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'learning_rate': 2e-5 * (0.95 ** epoch)
            }
            
            # Log to storage
            metrics_storage.log_metrics(
                run_id=run_id,
                task_id=current_task,
                epoch=epoch,
                step=step,
                metrics=train_metrics,
                metric_type="train"
            )
            
            print(f"    Epoch {epoch + 1}: train_acc={train_acc:.3f}, train_loss={train_loss:.3f}")
        
        # Final evaluation on current task
        final_accuracy = 0.6 + (i * 0.08) + np.random.normal(0, 0.02)
        continual_metrics.record_task_performance(current_task, current_task, final_accuracy)
        
        # Evaluate on all previous tasks (simulate forgetting)
        for j, prev_task in enumerate(tasks[:i]):
            # Simulate slight forgetting
            forgetting_factor = 0.98 - (i - j) * 0.02
            prev_accuracy = baseline_accuracies[prev_task] * forgetting_factor + np.random.normal(0, 0.01)
            
            continual_metrics.record_task_performance(current_task, prev_task, prev_accuracy)
            
            print(f"    {prev_task} accuracy: {prev_accuracy:.3f}")
    
    # Compute final metrics
    print(f"\n🎯 Computing continual learning metrics...")
    final_metrics = continual_metrics.compute_all_metrics()
    
    print(f"  ✅ Average Accuracy: {final_metrics.average_accuracy:.3f}")
    print(f"  ✅ Forgetting: {final_metrics.forgetting:.3f}")
    print(f"  ✅ Backward Transfer: {final_metrics.backward_transfer:.3f}")
    print(f"  ✅ Forward Transfer: {final_metrics.forward_transfer:.3f}")
    print(f"  ✅ Parameter Efficiency: {final_metrics.parameter_efficiency:.2f} acc/M params")
    print(f"  ✅ Memory Growth Rate: {final_metrics.memory_growth_rate:.2%}")
    
    # Show detailed metrics summary
    continual_metrics.print_summary()
    
    # Save metrics
    continual_metrics.save_metrics("./data_example/continual_metrics.json")
    print(f"\n💾 Saved continual learning metrics")
    
    # Export run metrics
    metrics_storage.export_metrics(run_id, "./data_example/run_metrics.json")
    print(f"💾 Exported run metrics")
    
    # Show run summary
    run_summary = metrics_storage.get_run_summary(run_id)
    print(f"\n📋 Run summary:")
    print(f"  - Total metrics logged: {run_summary['total_metrics']}")
    print(f"  - Tasks: {len(run_summary['tasks'])}")
    print(f"  - Final metrics per task:")
    for task_id, final_task_metrics in run_summary['final_metrics'].items():
        print(f"    - {task_id}: accuracy={final_task_metrics.get('accuracy', 0):.3f}")
    
    return metrics_storage, continual_metrics


def main():
    """Main demonstration of data management capabilities."""
    print("🚀 CONTINUAL TINY TRANSFORMER - DATA MANAGEMENT EXAMPLE")
    print("=" * 80)
    print("This example demonstrates comprehensive data management capabilities")
    print("including storage, processing, loading, and metrics tracking.")
    print("=" * 80)
    
    # Clean up any existing demo data
    import shutil
    demo_dir = Path("./data_example")
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    
    try:
        # 1. Advanced Data Processing
        text_processor, batch_processor, datasets = demonstrate_advanced_data_processing()
        
        # 2. Storage Management
        task_storage, checkpoint_manager, metrics_storage = demonstrate_storage_management(datasets)
        
        # 3. Continual Data Loading
        continual_loader, replay_loader, task_loaders = demonstrate_continual_data_loading(datasets, text_processor)
        
        # 4. Advanced Sampling
        sampler = demonstrate_advanced_sampling()
        
        # 5. Model Checkpointing
        checkpoint_manager = demonstrate_model_checkpointing()
        
        # 6. Metrics Tracking
        metrics_storage, continual_metrics = demonstrate_metrics_tracking()
        
        # Final Summary
        print("\n" + "="*80)
        print("🎉 DATA MANAGEMENT DEMONSTRATION COMPLETE!")
        print("="*80)
        
        print(f"\n✅ Successfully demonstrated:")
        print(f"  📄 Advanced text processing and tokenization")
        print(f"  💾 Persistent storage for tasks, models, and metrics")
        print(f"  🔄 Memory replay and continual data loading")
        print(f"  🎲 Advanced batch sampling strategies")
        print(f"  💾 Model checkpointing and versioning")
        print(f"  📊 Comprehensive metrics tracking and analysis")
        
        print(f"\n📁 Demo data saved to: {demo_dir.absolute()}")
        print(f"  - Task data: {demo_dir / 'tasks'}")
        print(f"  - Model checkpoints: {demo_dir / 'checkpoints'}")
        print(f"  - Metrics: {demo_dir / 'metrics'}")
        print(f"  - Continual learning metrics: continual_metrics.json")
        
        print(f"\n🎯 Key Features Demonstrated:")
        print(f"  🔹 Zero-parameter expansion with efficient data handling")
        print(f"  🔹 Robust storage and retrieval systems")
        print(f"  🔹 Advanced memory replay mechanisms")
        print(f"  🔹 Curriculum learning and task ordering")
        print(f"  🔹 Comprehensive performance tracking")
        print(f"  🔹 Production-ready data pipeline")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())