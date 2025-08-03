"""Data loaders for continual learning scenarios."""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional, Any, Iterator, Tuple, Union
import numpy as np
from pathlib import Path
import json
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class TaskDataset(Dataset):
    """Dataset wrapper for task-specific data."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        task_id: str,
        tokenizer,
        max_length: int = 512,
        label_mapping: Optional[Dict[str, int]] = None
    ):
        self.data = data
        self.task_id = task_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mapping = label_mapping or {}
        
        # Process and cache tokenized data
        self.processed_data = self._process_data()
        
        logger.info(f"Created TaskDataset for '{task_id}' with {len(self.data)} samples")
    
    def _process_data(self) -> List[Dict[str, torch.Tensor]]:
        """Process and tokenize all data."""
        processed = []
        
        for item in self.data:
            # Extract text and label
            text = item.get('text', '')
            label = item.get('label', 0)
            
            # Map label if mapping provided
            if isinstance(label, str) and label in self.label_mapping:
                label = self.label_mapping[label]
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            processed_item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long),
                'task_id': self.task_id
            }
            
            # Add any additional fields
            for key, value in item.items():
                if key not in ['text', 'label']:
                    processed_item[key] = value
            
            processed.append(processed_item)
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.processed_data[idx]
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in dataset."""
        distribution = defaultdict(int)
        for item in self.processed_data:
            label = item['labels'].item()
            distribution[label] += 1
        return dict(distribution)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        label_dist = self.get_label_distribution()
        
        # Text length statistics
        text_lengths = []
        for item in self.data:
            text_lengths.append(len(item.get('text', '').split()))
        
        return {
            'task_id': self.task_id,
            'num_samples': len(self.data),
            'num_classes': len(label_dist),
            'label_distribution': label_dist,
            'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0,
            'min_text_length': min(text_lengths) if text_lengths else 0
        }


class ContinualBatchSampler(Sampler):
    """Custom batch sampler for continual learning scenarios."""
    
    def __init__(
        self,
        dataset_sizes: List[int],
        batch_size: int,
        task_mixing_ratio: float = 0.0,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.task_mixing_ratio = task_mixing_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.total_size = sum(dataset_sizes)
        self.num_datasets = len(dataset_sizes)
        
        # Calculate dataset offsets
        self.dataset_offsets = [0]
        for size in dataset_sizes[:-1]:
            self.dataset_offsets.append(self.dataset_offsets[-1] + size)
    
    def __iter__(self) -> Iterator[List[int]]:
        # Create indices for each dataset
        dataset_indices = []
        for i, size in enumerate(self.dataset_sizes):
            indices = list(range(self.dataset_offsets[i], self.dataset_offsets[i] + size))
            if self.shuffle:
                random.shuffle(indices)
            dataset_indices.append(indices)
        
        # Generate batches
        while any(len(indices) >= self.batch_size for indices in dataset_indices):
            batch = []
            
            # Determine which datasets to sample from
            if self.task_mixing_ratio > 0 and self.num_datasets > 1:
                # Mixed batch
                primary_dataset = random.randint(0, self.num_datasets - 1)
                primary_size = int(self.batch_size * (1 - self.task_mixing_ratio))
                
                # Sample from primary dataset
                if len(dataset_indices[primary_dataset]) >= primary_size:
                    batch.extend(dataset_indices[primary_dataset][:primary_size])
                    dataset_indices[primary_dataset] = dataset_indices[primary_dataset][primary_size:]
                
                # Sample from other datasets
                remaining_size = self.batch_size - len(batch)
                other_datasets = [i for i in range(self.num_datasets) if i != primary_dataset]
                
                for dataset_idx in other_datasets:
                    if remaining_size <= 0:
                        break
                    
                    available = len(dataset_indices[dataset_idx])
                    if available > 0:
                        take = min(remaining_size, available)
                        batch.extend(dataset_indices[dataset_idx][:take])
                        dataset_indices[dataset_idx] = dataset_indices[dataset_idx][take:]
                        remaining_size -= take
            
            else:
                # Pure batch from one dataset
                for i, indices in enumerate(dataset_indices):
                    if len(indices) >= self.batch_size:
                        batch = indices[:self.batch_size]
                        dataset_indices[i] = indices[self.batch_size:]
                        break
            
            if len(batch) == self.batch_size or (not self.drop_last and batch):
                yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return self.total_size // self.batch_size
        else:
            return (self.total_size + self.batch_size - 1) // self.batch_size


class ContinualDataLoader:
    """Data loader manager for continual learning."""
    
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        # Task datasets
        self.task_datasets: Dict[str, TaskDataset] = {}
        self.task_loaders: Dict[str, DataLoader] = {}
        
        # Current task tracking
        self.current_task = None
        self.task_sequence = []
        
    def add_task_dataset(
        self,
        task_id: str,
        dataset: TaskDataset,
        train_ratio: float = 0.8,
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """Add a task dataset and create train/val loaders."""
        
        self.task_datasets[task_id] = dataset
        if task_id not in self.task_sequence:
            self.task_sequence.append(task_id)
        
        # Split into train/validation
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible splits
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self._collate_fn
        )
        
        # Store loaders
        self.task_loaders[f"{task_id}_train"] = train_loader
        self.task_loaders[f"{task_id}_val"] = val_loader
        
        logger.info(f"Added task dataset '{task_id}': {train_size} train, {val_size} val samples")
        
        return train_loader, val_loader
    
    def get_task_loader(self, task_id: str, split: str = "train") -> Optional[DataLoader]:
        """Get data loader for specific task and split."""
        key = f"{task_id}_{split}"
        return self.task_loaders.get(key)
    
    def get_multi_task_loader(
        self,
        task_ids: List[str],
        split: str = "train",
        mixing_ratio: float = 0.2
    ) -> DataLoader:
        """Create a multi-task data loader."""
        
        # Collect datasets
        datasets = []
        for task_id in task_ids:
            loader = self.get_task_loader(task_id, split)
            if loader is not None:
                datasets.append(loader.dataset)
        
        if not datasets:
            raise ValueError("No valid datasets found for specified tasks")
        
        # Combine datasets
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        
        # Create custom sampler for task mixing
        dataset_sizes = [len(d) for d in datasets]
        sampler = ContinualBatchSampler(
            dataset_sizes=dataset_sizes,
            batch_size=self.batch_size,
            task_mixing_ratio=mixing_ratio,
            shuffle=(split == "train")
        )
        
        return DataLoader(
            combined_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def get_continual_loader(self, current_task_id: str, replay_tasks: List[str] = None) -> DataLoader:
        """Get loader for continual learning with replay."""
        if replay_tasks is None:
            replay_tasks = []
        
        all_tasks = [current_task_id] + replay_tasks
        return self.get_multi_task_loader(all_tasks, "train", mixing_ratio=0.3)
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        if not batch:
            return {}
        
        # Stack tensors
        collated = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch])
            elif key == 'task_id':
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = [item[key] for item in batch]
        
        return collated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        stats = {
            'num_tasks': len(self.task_datasets),
            'task_sequence': self.task_sequence,
            'total_samples': sum(len(dataset) for dataset in self.task_datasets.values()),
            'task_statistics': {}
        }
        
        for task_id, dataset in self.task_datasets.items():
            stats['task_statistics'][task_id] = dataset.get_statistics()
        
        return stats


class MemoryReplayDataLoader:
    """Specialized data loader for memory replay in continual learning."""
    
    def __init__(
        self,
        buffer_size: int = 1000,
        replay_ratio: float = 0.2,
        batch_size: int = 16,
        strategy: str = "random"  # random, balanced, importance
    ):
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.batch_size = batch_size
        self.strategy = strategy
        
        # Memory buffers per task
        self.memory_buffers: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        self.buffer_indices: Dict[str, int] = {}
        self.importance_scores: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized MemoryReplayDataLoader with {buffer_size} buffer size")
    
    def add_to_memory(
        self,
        task_id: str,
        batch: Dict[str, torch.Tensor],
        importance_scores: Optional[List[float]] = None
    ):
        """Add batch to memory buffer."""
        if task_id not in self.memory_buffers:
            self.memory_buffers[task_id] = []
            self.buffer_indices[task_id] = 0
            self.importance_scores[task_id] = []
        
        buffer = self.memory_buffers[task_id]
        batch_size = batch['input_ids'].size(0)
        
        for i in range(batch_size):
            # Extract single sample
            sample = {
                key: value[i:i+1].clone().detach() if isinstance(value, torch.Tensor) else value[i]
                for key, value in batch.items()
            }
            
            # Add importance score
            importance = importance_scores[i] if importance_scores else 1.0
            
            if len(buffer) < self.buffer_size:
                buffer.append(sample)
                self.importance_scores[task_id].append(importance)
            else:
                # Replace sample based on strategy
                replace_idx = self._get_replacement_index(task_id, importance)
                buffer[replace_idx] = sample
                self.importance_scores[task_id][replace_idx] = importance
    
    def _get_replacement_index(self, task_id: str, new_importance: float) -> int:
        """Get index to replace based on strategy."""
        buffer_size = len(self.memory_buffers[task_id])
        
        if self.strategy == "random":
            return random.randint(0, buffer_size - 1)
        
        elif self.strategy == "importance":
            # Replace least important sample
            importance_scores = self.importance_scores[task_id]
            return np.argmin(importance_scores)
        
        elif self.strategy == "balanced":
            # Try to maintain balanced classes
            # For simplicity, use round-robin replacement
            idx = self.buffer_indices[task_id]
            self.buffer_indices[task_id] = (idx + 1) % buffer_size
            return idx
        
        else:
            return random.randint(0, buffer_size - 1)
    
    def sample_replay_batch(
        self,
        exclude_task: Optional[str] = None,
        num_samples: Optional[int] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a replay batch from memory."""
        
        # Get available tasks
        available_tasks = [
            task_id for task_id in self.memory_buffers.keys()
            if task_id != exclude_task and self.memory_buffers[task_id]
        ]
        
        if not available_tasks:
            return None
        
        # Calculate number of samples
        if num_samples is None:
            num_samples = max(1, int(self.batch_size * self.replay_ratio))
        
        # Sample from tasks
        samples = []
        samples_per_task = max(1, num_samples // len(available_tasks))
        
        for task_id in available_tasks:
            buffer = self.memory_buffers[task_id]
            task_samples = min(samples_per_task, len(buffer))
            
            if self.strategy == "importance":
                # Sample based on importance scores
                importance_scores = np.array(self.importance_scores[task_id])
                probabilities = importance_scores / importance_scores.sum()
                indices = np.random.choice(
                    len(buffer), 
                    size=task_samples, 
                    replace=False,
                    p=probabilities
                )
            else:
                # Random sampling
                indices = random.sample(range(len(buffer)), task_samples)
            
            for idx in indices:
                samples.append(buffer[idx])
            
            if len(samples) >= num_samples:
                break
        
        if not samples:
            return None
        
        # Collate samples into batch
        batch = {}
        for key in samples[0].keys():
            if isinstance(samples[0][key], torch.Tensor):
                batch[key] = torch.cat([sample[key] for sample in samples], dim=0)
            else:
                batch[key] = [sample[key] for sample in samples]
        
        return batch
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory buffer statistics."""
        stats = {
            'buffer_size': self.buffer_size,
            'replay_ratio': self.replay_ratio,
            'strategy': self.strategy,
            'total_tasks': len(self.memory_buffers),
            'task_buffers': {}
        }
        
        for task_id, buffer in self.memory_buffers.items():
            stats['task_buffers'][task_id] = {
                'samples': len(buffer),
                'utilization': len(buffer) / self.buffer_size,
                'avg_importance': np.mean(self.importance_scores[task_id]) if self.importance_scores[task_id] else 0.0
            }
        
        return stats
    
    def clear_memory(self, task_id: Optional[str] = None):
        """Clear memory buffers."""
        if task_id is None:
            self.memory_buffers.clear()
            self.buffer_indices.clear()
            self.importance_scores.clear()
        else:
            if task_id in self.memory_buffers:
                del self.memory_buffers[task_id]
                del self.buffer_indices[task_id]
                del self.importance_scores[task_id]
        
        logger.info(f"Cleared memory for task: {task_id or 'all tasks'}")


def create_synthetic_task_data(
    task_id: str,
    num_samples: int = 1000,
    num_classes: int = 3,
    vocab_size: int = 1000,
    seq_length: int = 64,
    class_names: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Create synthetic data for testing continual learning."""
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    data = []
    for i in range(num_samples):
        # Generate synthetic text (token IDs as string)
        token_ids = np.random.randint(100, vocab_size, size=seq_length)
        text = " ".join(map(str, token_ids))
        
        # Random label
        label = random.randint(0, num_classes - 1)
        
        data.append({
            'text': text,
            'label': label,
            'label_name': class_names[label],
            'task_id': task_id,
            'sample_id': i
        })
    
    return data