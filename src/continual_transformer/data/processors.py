"""Data processors for continual learning."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import AutoTokenizer
import logging
from collections import defaultdict, Counter
import re
import string

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text preprocessing and tokenization for continual learning."""
    
    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir
        )
        
        # Add special tokens if needed
        self._add_special_tokens()
        
        # Text statistics
        self.vocab_stats = {}
        self.length_stats = {}
        
        logger.info(f"Initialized TextProcessor with {tokenizer_name}")
    
    def _add_special_tokens(self):
        """Add task-specific special tokens."""
        special_tokens = {
            'additional_special_tokens': ['[TASK]', '[SEP_TASK]', '[REPLAY]']
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            logger.info(f"Added {num_added} special tokens")
    
    def preprocess_text(self, text: str, task_id: Optional[str] = None) -> str:
        """Preprocess text before tokenization."""
        
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Optional: Add task identifier
        if task_id:
            text = f"[TASK] {task_id} [SEP_TASK] {text}"
        
        return text
    
    def tokenize_batch(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        task_id: Optional[str] = None,
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding: Union[bool, str] = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        
        # Preprocess texts
        processed_texts = [
            self.preprocess_text(text, task_id) for text in texts
        ]
        
        # Tokenize
        encoding = self.tokenizer(
            processed_texts,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            max_length=self.max_length,
            return_tensors=return_tensors,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        # Add labels if provided
        if labels is not None:
            encoding['labels'] = torch.tensor(labels, dtype=torch.long)
        
        # Add task information
        if task_id is not None:
            encoding['task_ids'] = [task_id] * len(texts)
        
        # Update statistics
        self._update_statistics(processed_texts, task_id)
        
        return encoding
    
    def tokenize_single(
        self,
        text: str,
        task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a single text."""
        return self.tokenize_batch([text], task_id=task_id, **kwargs)
    
    def decode_batch(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode batch of token IDs back to text."""
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad': self.tokenizer.pad_token_id,
            'cls': self.tokenizer.cls_token_id,
            'sep': self.tokenizer.sep_token_id,
            'unk': self.tokenizer.unk_token_id,
            'mask': self.tokenizer.mask_token_id
        }
    
    def _update_statistics(self, texts: List[str], task_id: Optional[str] = None):
        """Update text statistics."""
        if task_id is None:
            task_id = "default"
        
        if task_id not in self.vocab_stats:
            self.vocab_stats[task_id] = Counter()
            self.length_stats[task_id] = []
        
        for text in texts:
            # Update vocabulary statistics
            tokens = self.tokenizer.tokenize(text)
            self.vocab_stats[task_id].update(tokens)
            
            # Update length statistics
            self.length_stats[task_id].append(len(tokens))
    
    def get_statistics(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get text processing statistics."""
        if task_id is None:
            # Aggregate statistics across all tasks
            all_vocab = Counter()
            all_lengths = []
            
            for vocab_counter in self.vocab_stats.values():
                all_vocab.update(vocab_counter)
            
            for lengths in self.length_stats.values():
                all_lengths.extend(lengths)
            
            return {
                'total_tasks': len(self.vocab_stats),
                'unique_tokens': len(all_vocab),
                'total_tokens': sum(all_vocab.values()),
                'avg_length': np.mean(all_lengths) if all_lengths else 0,
                'max_length': max(all_lengths) if all_lengths else 0,
                'min_length': min(all_lengths) if all_lengths else 0,
                'most_common_tokens': all_vocab.most_common(10)
            }
        else:
            # Task-specific statistics
            if task_id not in self.vocab_stats:
                return {}
            
            vocab = self.vocab_stats[task_id]
            lengths = self.length_stats[task_id]
            
            return {
                'task_id': task_id,
                'unique_tokens': len(vocab),
                'total_tokens': sum(vocab.values()),
                'avg_length': np.mean(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0,
                'min_length': min(lengths) if lengths else 0,
                'most_common_tokens': vocab.most_common(10)
            }


class TaskBatchProcessor:
    """Processes batches for multi-task continual learning."""
    
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.task_label_mappings: Dict[str, Dict[str, int]] = {}
        self.task_stats = defaultdict(lambda: defaultdict(int))
    
    def register_task_labels(self, task_id: str, label_mapping: Dict[str, int]):
        """Register label mapping for a task."""
        self.task_label_mappings[task_id] = label_mapping
        logger.info(f"Registered labels for task '{task_id}': {label_mapping}")
    
    def process_task_batch(
        self,
        batch_data: List[Dict[str, Any]],
        task_id: str
    ) -> Dict[str, torch.Tensor]:
        """Process a batch of data for a specific task."""
        
        # Extract texts and labels
        texts = [item['text'] for item in batch_data]
        labels = []
        
        label_mapping = self.task_label_mappings.get(task_id, {})
        
        for item in batch_data:
            label = item.get('label', 0)
            
            # Map string labels to integers if needed
            if isinstance(label, str) and label in label_mapping:
                label = label_mapping[label]
            
            labels.append(label)
        
        # Tokenize batch
        processed_batch = self.text_processor.tokenize_batch(
            texts=texts,
            labels=labels,
            task_id=task_id
        )
        
        # Update statistics
        self.task_stats[task_id]['num_batches'] += 1
        self.task_stats[task_id]['total_samples'] += len(batch_data)
        
        # Add additional metadata
        processed_batch['task_id'] = task_id
        processed_batch['batch_size'] = len(batch_data)
        
        return processed_batch
    
    def process_multi_task_batch(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Process a batch containing data from multiple tasks."""
        
        # Group by task
        task_groups = defaultdict(list)
        for item in batch_data:
            task_id = item.get('task_id', 'unknown')
            task_groups[task_id].append(item)
        
        # Process each task group
        all_inputs = []
        all_labels = []
        all_attention_masks = []
        task_ids = []
        
        for task_id, task_items in task_groups.items():
            task_batch = self.process_task_batch(task_items, task_id)
            
            all_inputs.append(task_batch['input_ids'])
            all_labels.append(task_batch['labels'])
            all_attention_masks.append(task_batch['attention_mask'])
            task_ids.extend([task_id] * len(task_items))
        
        # Concatenate all task batches
        combined_batch = {
            'input_ids': torch.cat(all_inputs, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'task_ids': task_ids,
            'num_tasks': len(task_groups),
            'batch_size': len(batch_data)
        }
        
        return combined_batch
    
    def create_balanced_batch(
        self,
        data_by_task: Dict[str, List[Dict[str, Any]]],
        batch_size: int,
        task_sampling_strategy: str = "uniform"
    ) -> Dict[str, torch.Tensor]:
        """Create a balanced batch across multiple tasks."""
        
        available_tasks = [
            task_id for task_id, data in data_by_task.items() 
            if len(data) > 0
        ]
        
        if not available_tasks:
            raise ValueError("No data available for any task")
        
        # Determine samples per task
        if task_sampling_strategy == "uniform":
            samples_per_task = batch_size // len(available_tasks)
            remaining_samples = batch_size % len(available_tasks)
        else:
            raise NotImplementedError(f"Strategy '{task_sampling_strategy}' not implemented")
        
        # Sample from each task
        batch_items = []
        
        for i, task_id in enumerate(available_tasks):
            task_data = data_by_task[task_id]
            num_samples = samples_per_task
            
            # Distribute remaining samples
            if i < remaining_samples:
                num_samples += 1
            
            # Random sampling
            if len(task_data) >= num_samples:
                sampled_items = np.random.choice(
                    task_data, 
                    size=num_samples, 
                    replace=False
                ).tolist()
            else:
                # With replacement if not enough samples
                sampled_items = np.random.choice(
                    task_data, 
                    size=num_samples, 
                    replace=True
                ).tolist()
            
            batch_items.extend(sampled_items)
        
        # Process the balanced batch
        return self.process_multi_task_batch(batch_items)
    
    def get_task_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get processing statistics for all tasks."""
        stats = {}
        
        for task_id, task_stats in self.task_stats.items():
            text_stats = self.text_processor.get_statistics(task_id)
            
            stats[task_id] = {
                'processing_stats': dict(task_stats),
                'text_stats': text_stats,
                'label_mapping': self.task_label_mappings.get(task_id, {}),
                'num_classes': len(self.task_label_mappings.get(task_id, {}))
            }
        
        return stats


class ContinualBatchSampler:
    """Advanced batch sampler for continual learning scenarios."""
    
    def __init__(
        self,
        datasets: Dict[str, Any],
        batch_size: int = 16,
        replay_ratio: float = 0.2,
        curriculum_strategy: str = "none"  # none, difficulty, similarity
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self.curriculum_strategy = curriculum_strategy
        
        # Task ordering and curriculum
        self.task_order = list(datasets.keys())
        self.task_difficulties = {}
        self.task_similarities = {}
        
        # Replay buffer management
        self.replay_buffers = {}
        self.replay_indices = {}
        
        logger.info(f"Initialized ContinualBatchSampler with {len(datasets)} tasks")
    
    def set_task_difficulties(self, difficulties: Dict[str, float]):
        """Set task difficulties for curriculum learning."""
        self.task_difficulties = difficulties
        
        if self.curriculum_strategy == "difficulty":
            # Reorder tasks by difficulty
            self.task_order = sorted(
                self.task_order,
                key=lambda x: difficulties.get(x, 0.5)
            )
            logger.info(f"Reordered tasks by difficulty: {self.task_order}")
    
    def set_task_similarities(self, similarities: Dict[str, Dict[str, float]]):
        """Set task similarity matrix for curriculum learning."""
        self.task_similarities = similarities
        
        if self.curriculum_strategy == "similarity":
            # Reorder tasks by similarity (simple greedy approach)
            ordered_tasks = [self.task_order[0]]  # Start with first task
            remaining_tasks = set(self.task_order[1:])
            
            while remaining_tasks:
                current_task = ordered_tasks[-1]
                
                # Find most similar remaining task
                best_task = None
                best_similarity = -1
                
                for task in remaining_tasks:
                    similarity = similarities.get(current_task, {}).get(task, 0)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_task = task
                
                if best_task:
                    ordered_tasks.append(best_task)
                    remaining_tasks.remove(best_task)
                else:
                    # Fallback: add any remaining task
                    ordered_tasks.append(remaining_tasks.pop())
            
            self.task_order = ordered_tasks
            logger.info(f"Reordered tasks by similarity: {self.task_order}")
    
    def sample_continual_batch(
        self,
        current_task: str,
        current_data: List[Dict[str, Any]],
        include_replay: bool = True
    ) -> List[Dict[str, Any]]:
        """Sample a batch for continual learning."""
        
        # Calculate batch composition
        current_task_samples = int(self.batch_size * (1 - self.replay_ratio))
        replay_samples = self.batch_size - current_task_samples
        
        batch_items = []
        
        # Sample from current task
        if len(current_data) >= current_task_samples:
            current_batch = np.random.choice(
                current_data,
                size=current_task_samples,
                replace=False
            ).tolist()
        else:
            current_batch = np.random.choice(
                current_data,
                size=current_task_samples,
                replace=True
            ).tolist()
        
        batch_items.extend(current_batch)
        
        # Add replay samples if enabled
        if include_replay and replay_samples > 0 and self.replay_buffers:
            # Sample from previous tasks
            available_tasks = [
                task for task in self.replay_buffers.keys()
                if task != current_task and self.replay_buffers[task]
            ]
            
            if available_tasks:
                # Distribute replay samples across available tasks
                samples_per_task = max(1, replay_samples // len(available_tasks))
                
                for task_id in available_tasks:
                    if replay_samples <= 0:
                        break
                    
                    task_buffer = self.replay_buffers[task_id]
                    task_samples = min(samples_per_task, replay_samples, len(task_buffer))
                    
                    replay_batch = np.random.choice(
                        task_buffer,
                        size=task_samples,
                        replace=False
                    ).tolist()
                    
                    # Mark as replay samples
                    for item in replay_batch:
                        item['is_replay'] = True
                    
                    batch_items.extend(replay_batch)
                    replay_samples -= task_samples
        
        return batch_items
    
    def update_replay_buffer(
        self,
        task_id: str,
        data: List[Dict[str, Any]],
        buffer_size: int = 1000
    ):
        """Update replay buffer for a task."""
        
        if task_id not in self.replay_buffers:
            self.replay_buffers[task_id] = []
            self.replay_indices[task_id] = 0
        
        buffer = self.replay_buffers[task_id]
        
        # Add data to buffer
        for item in data:
            if len(buffer) < buffer_size:
                buffer.append(item.copy())
            else:
                # Replace oldest item (circular buffer)
                buffer[self.replay_indices[task_id]] = item.copy()
                self.replay_indices[task_id] = (self.replay_indices[task_id] + 1) % buffer_size
        
        logger.info(f"Updated replay buffer for '{task_id}': {len(buffer)} samples")
    
    def get_curriculum_order(self) -> List[str]:
        """Get task order based on curriculum strategy."""
        return self.task_order.copy()
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about batch sampling."""
        return {
            'batch_size': self.batch_size,
            'replay_ratio': self.replay_ratio,
            'curriculum_strategy': self.curriculum_strategy,
            'task_order': self.task_order,
            'replay_buffer_sizes': {
                task: len(buffer) for task, buffer in self.replay_buffers.items()
            },
            'task_difficulties': self.task_difficulties,
            'num_similarity_pairs': sum(
                len(similarities) for similarities in self.task_similarities.values()
            )
        }