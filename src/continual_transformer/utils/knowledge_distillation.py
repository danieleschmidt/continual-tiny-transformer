"""Knowledge distillation utilities for preventing catastrophic forgetting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class KnowledgeDistillation:
    """Knowledge distillation for continual learning."""
    
    def __init__(self, config):
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.knowledge_distillation_alpha
    
    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """Compute knowledge distillation loss.
        
        Args:
            student_logits: Current model predictions [batch_size, num_classes]
            teacher_logits: Previous model predictions [batch_size, num_classes]
            temperature: Temperature for softmax (higher = softer)
            alpha: Weight for distillation loss
            
        Returns:
            Knowledge distillation loss
        """
        if teacher_logits is None:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        temp = temperature or self.temperature
        alpha = alpha or self.alpha
        
        # Ensure same device and shape
        teacher_logits = teacher_logits.to(student_logits.device)
        
        # Handle shape mismatch (different number of classes)
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            min_classes = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_logits = student_logits[:, :min_classes]
            teacher_logits = teacher_logits[:, :min_classes]
        
        # Compute soft targets using temperature
        student_soft = F.log_softmax(student_logits / temp, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (temp ** 2)
        
        return alpha * distillation_loss
    
    def compute_feature_distillation_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """Compute feature-level distillation loss.
        
        Args:
            student_features: Student model features [batch_size, seq_len, hidden_size]
            teacher_features: Teacher model features [batch_size, seq_len, hidden_size]
            loss_type: Type of loss ('mse', 'cosine', 'attention')
            
        Returns:
            Feature distillation loss
        """
        # Ensure same device
        teacher_features = teacher_features.to(student_features.device)
        
        if loss_type == "mse":
            # Mean squared error between features
            loss = F.mse_loss(student_features, teacher_features)
            
        elif loss_type == "cosine":
            # Cosine similarity loss
            student_norm = F.normalize(student_features, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
            loss = 1 - F.cosine_similarity(student_norm, teacher_norm, dim=-1).mean()
            
        elif loss_type == "attention":
            # Attention transfer loss
            loss = self._attention_transfer_loss(student_features, teacher_features)
            
        else:
            raise ValueError(f"Unknown feature distillation loss type: {loss_type}")
        
        return loss
    
    def _attention_transfer_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention transfer loss between student and teacher."""
        # Compute attention maps
        student_attention = self._compute_attention_map(student_features)
        teacher_attention = self._compute_attention_map(teacher_features)
        
        # MSE loss between attention maps
        return F.mse_loss(student_attention, teacher_attention)
    
    def _compute_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """Compute attention map from features."""
        # Simple attention: dot product between all pairs
        batch_size, seq_len, hidden_size = features.shape
        
        # Reshape for batch matrix multiplication
        features_flat = features.view(batch_size, seq_len, hidden_size)
        
        # Compute attention scores
        attention_scores = torch.bmm(features_flat, features_flat.transpose(1, 2))
        
        # Normalize
        attention_scores = attention_scores / (hidden_size ** 0.5)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights


class ProgressiveKnowledgeDistillation:
    """Progressive knowledge distillation that adapts over tasks."""
    
    def __init__(self, config):
        self.config = config
        self.task_teachers: Dict[str, Dict[str, torch.Tensor]] = {}
        self.distillation_weights: Dict[str, float] = {}
        self.task_similarities: Dict[str, Dict[str, float]] = {}
    
    def add_teacher(self, task_id: str, model_state: Dict[str, torch.Tensor]):
        """Add a teacher model for a specific task."""
        self.task_teachers[task_id] = model_state
        self.distillation_weights[task_id] = 1.0
        logger.info(f"Added teacher for task '{task_id}'")
    
    def compute_multi_teacher_loss(
        self,
        student_logits: torch.Tensor,
        current_task_id: str,
        similarity_threshold: float = 0.5
    ) -> torch.Tensor:
        """Compute loss from multiple relevant teachers."""
        if not self.task_teachers:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        total_weight = 0.0
        
        kd = KnowledgeDistillation(self.config)
        
        for teacher_task_id, teacher_state in self.task_teachers.items():
            if teacher_task_id == current_task_id:
                continue
            
            # Get similarity weight
            similarity = self.get_task_similarity(current_task_id, teacher_task_id)
            
            if similarity < similarity_threshold:
                continue
            
            # Get teacher predictions (this would need to be implemented based on your model)
            teacher_logits = self._get_teacher_predictions(teacher_state, student_logits)
            
            if teacher_logits is not None:
                # Compute weighted distillation loss
                teacher_loss = kd.compute_loss(student_logits, teacher_logits)
                weight = similarity * self.distillation_weights[teacher_task_id]
                
                total_loss = total_loss + weight * teacher_loss
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        return total_loss
    
    def get_task_similarity(self, task1: str, task2: str) -> float:
        """Get similarity between two tasks."""
        if task1 in self.task_similarities and task2 in self.task_similarities[task1]:
            return self.task_similarities[task1][task2]
        
        # Default similarity (could be computed from task embeddings, etc.)
        return 0.5
    
    def set_task_similarity(self, task1: str, task2: str, similarity: float):
        """Set similarity between two tasks."""
        if task1 not in self.task_similarities:
            self.task_similarities[task1] = {}
        if task2 not in self.task_similarities:
            self.task_similarities[task2] = {}
        
        self.task_similarities[task1][task2] = similarity
        self.task_similarities[task2][task1] = similarity
    
    def update_distillation_weights(self, performance_scores: Dict[str, float]):
        """Update distillation weights based on task performance."""
        for task_id, score in performance_scores.items():
            if task_id in self.distillation_weights:
                # Higher performing tasks get higher weights
                self.distillation_weights[task_id] = score
        
        logger.info("Updated distillation weights based on performance")
    
    def _get_teacher_predictions(
        self,
        teacher_state: Dict[str, torch.Tensor],
        student_input: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get predictions from teacher model (placeholder)."""
        # This would need to be implemented based on your specific model architecture
        # For now, return None to indicate no teacher predictions available
        return None


class MemoryReplay:
    """Memory replay for continual learning."""
    
    def __init__(self, config):
        self.config = config
        self.buffer_size = config.replay_buffer_size
        self.replay_ratio = config.replay_ratio
        
        # Memory buffers per task
        self.memory_buffers: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        self.buffer_indices: Dict[str, int] = {}
    
    def add_to_memory(self, task_id: str, batch: Dict[str, torch.Tensor]):
        """Add batch to memory buffer for a task."""
        if task_id not in self.memory_buffers:
            self.memory_buffers[task_id] = []
            self.buffer_indices[task_id] = 0
        
        buffer = self.memory_buffers[task_id]
        
        # Extract individual samples from batch
        batch_size = batch['input_ids'].size(0)
        
        for i in range(batch_size):
            sample = {
                key: value[i:i+1].clone().detach()  # Keep batch dimension
                for key, value in batch.items()
            }
            
            if len(buffer) < self.buffer_size:
                buffer.append(sample)
            else:
                # Replace oldest sample (circular buffer)
                buffer[self.buffer_indices[task_id]] = sample
                self.buffer_indices[task_id] = (self.buffer_indices[task_id] + 1) % self.buffer_size
    
    def sample_from_memory(self, task_id: str, num_samples: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample from memory buffer for a task."""
        if task_id not in self.memory_buffers or not self.memory_buffers[task_id]:
            return None
        
        buffer = self.memory_buffers[task_id]
        num_samples = min(num_samples, len(buffer))
        
        # Random sampling
        indices = torch.randperm(len(buffer))[:num_samples]
        
        # Combine samples into batch
        batched_samples = {}
        for key in buffer[0].keys():
            batched_samples[key] = torch.cat([
                buffer[idx][key] for idx in indices
            ], dim=0)
        
        return batched_samples
    
    def get_replay_batch(self, current_task_id: str, current_batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get replay batches from previous tasks."""
        replay_batches = []
        
        # Calculate replay samples per task
        num_replay_samples = int(current_batch_size * self.replay_ratio)
        
        if num_replay_samples == 0:
            return replay_batches
        
        # Get samples from all previous tasks
        previous_tasks = [task_id for task_id in self.memory_buffers.keys() 
                         if task_id != current_task_id]
        
        if not previous_tasks:
            return replay_batches
        
        samples_per_task = max(1, num_replay_samples // len(previous_tasks))
        
        for task_id in previous_tasks:
            replay_batch = self.sample_from_memory(task_id, samples_per_task)
            if replay_batch is not None:
                replay_batch['task_id'] = task_id
                replay_batches.append(replay_batch)
        
        return replay_batches
    
    def clear_memory(self, task_id: Optional[str] = None):
        """Clear memory buffer(s)."""
        if task_id is None:
            self.memory_buffers.clear()
            self.buffer_indices.clear()
            logger.info("Cleared all memory buffers")
        else:
            if task_id in self.memory_buffers:
                del self.memory_buffers[task_id]
                del self.buffer_indices[task_id]
                logger.info(f"Cleared memory buffer for task '{task_id}'")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory buffer statistics."""
        stats = {
            "total_tasks": len(self.memory_buffers),
            "buffer_size_limit": self.buffer_size,
            "replay_ratio": self.replay_ratio,
            "task_buffers": {}
        }
        
        for task_id, buffer in self.memory_buffers.items():
            stats["task_buffers"][task_id] = {
                "samples": len(buffer),
                "utilization": len(buffer) / self.buffer_size
            }
        
        return stats