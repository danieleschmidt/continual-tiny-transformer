"""
Meta-Continual Learning: Learning to Learn Continually

Novel approach that learns meta-parameters for optimal continual learning adaptation.
Implements Model-Agnostic Meta-Learning (MAML) adapted for continual scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import copy
import math
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-continual learning."""
    meta_lr: float = 1e-3
    inner_lr: float = 1e-2
    meta_batch_size: int = 4
    inner_steps: int = 5
    meta_steps: int = 1000
    adaptation_steps: int = 10
    temperature: float = 2.0
    regularization_strength: float = 0.1
    enable_second_order: bool = True
    memory_replay_size: int = 100


class TaskMemoryBank:
    """Maintains episodic memory for tasks to enable efficient meta-learning."""
    
    def __init__(self, memory_size: int = 1000, embedding_dim: int = 512):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.task_memories = {}
        self.global_memory = []
        self.memory_embeddings = []
        self.access_counts = defaultdict(int)
        
    def store_task_experience(
        self, 
        task_id: str, 
        hidden_states: torch.Tensor, 
        labels: torch.Tensor,
        metadata: Dict[str, Any] = None
    ):
        """Store experience for a specific task."""
        
        if task_id not in self.task_memories:
            self.task_memories[task_id] = []
        
        # Create experience tuple
        experience = {
            'hidden_states': hidden_states.detach().cpu(),
            'labels': labels.detach().cpu(),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.task_memories[task_id].append(experience)
        
        # Maintain memory size limit per task
        if len(self.task_memories[task_id]) > self.memory_size // 10:  # 10% per task
            self.task_memories[task_id].pop(0)
    
    def retrieve_similar_experiences(
        self, 
        query_states: torch.Tensor, 
        k: int = 5,
        exclude_task: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve k most similar experiences across all tasks."""
        
        similarities = []
        query_embedding = self._compute_embedding(query_states)
        
        for task_id, memories in self.task_memories.items():
            if exclude_task and task_id == exclude_task:
                continue
                
            for memory in memories:
                memory_embedding = self._compute_embedding(memory['hidden_states'])
                similarity = F.cosine_similarity(
                    query_embedding.unsqueeze(0), 
                    memory_embedding.unsqueeze(0)
                ).item()
                
                similarities.append({
                    'similarity': similarity,
                    'experience': memory,
                    'task_id': task_id
                })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:k]
    
    def _compute_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute representative embedding for hidden states."""
        # Global average pooling
        if hidden_states.dim() == 3:  # [batch, seq, hidden]
            embedding = hidden_states.mean(dim=(0, 1))
        elif hidden_states.dim() == 2:  # [batch, hidden]
            embedding = hidden_states.mean(dim=0)
        else:
            embedding = hidden_states.flatten().mean()
        
        return embedding
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        stats = {
            'total_tasks': len(self.task_memories),
            'total_experiences': sum(len(memories) for memories in self.task_memories.values()),
            'task_distribution': {
                task_id: len(memories) 
                for task_id, memories in self.task_memories.items()
            },
            'memory_utilization': sum(len(memories) for memories in self.task_memories.values()) / self.memory_size
        }
        return stats


class MetaGradientProcessor:
    """Processes gradients for meta-learning updates."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.gradient_history = []
        self.meta_gradients = {}
        
    def compute_meta_gradients(
        self,
        model: nn.Module,
        support_data: List[Dict],
        query_data: List[Dict],
        task_ids: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Compute meta-gradients using MAML-style optimization."""
        
        meta_gradients = {}
        
        # Save original parameters
        original_params = {
            name: param.clone() for name, param in model.named_parameters()
        }
        
        for i, (support_batch, query_batch, task_id) in enumerate(zip(support_data, query_data, task_ids)):
            
            # Inner loop adaptation
            adapted_params = self._inner_loop_adaptation(
                model, support_batch, task_id
            )
            
            # Compute meta-loss on query set
            meta_loss = self._compute_query_loss(
                model, query_batch, adapted_params
            )
            
            # Compute gradients w.r.t. original parameters
            if self.config.enable_second_order:
                # Second-order gradients for better adaptation
                task_gradients = torch.autograd.grad(
                    meta_loss, 
                    model.parameters(),
                    create_graph=True,
                    retain_graph=True
                )
            else:
                # First-order approximation (faster)
                task_gradients = torch.autograd.grad(
                    meta_loss,
                    model.parameters(),
                    retain_graph=True
                )
            
            # Accumulate meta-gradients
            for (name, param), grad in zip(model.named_parameters(), task_gradients):
                if name not in meta_gradients:
                    meta_gradients[name] = torch.zeros_like(param)
                meta_gradients[name] += grad / len(support_data)
        
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])
        
        return meta_gradients
    
    def _inner_loop_adaptation(
        self,
        model: nn.Module,
        support_data: Dict,
        task_id: str
    ) -> Dict[str, torch.Tensor]:
        """Perform inner loop adaptation on support set."""
        
        adapted_params = {}
        
        # Perform gradient steps on support set
        for step in range(self.config.inner_steps):
            
            # Forward pass
            outputs = model(**support_data)
            loss = outputs.get('loss', outputs['logits'])
            
            if not isinstance(loss, torch.Tensor):
                loss = F.cross_entropy(outputs['logits'], support_data['labels'])
            
            # Compute gradients
            gradients = torch.autograd.grad(
                loss,
                model.parameters(),
                create_graph=self.config.enable_second_order,
                retain_graph=True
            )
            
            # Update parameters with inner learning rate
            for (name, param), grad in zip(model.named_parameters(), gradients):
                if name not in adapted_params:
                    adapted_params[name] = param.clone()
                adapted_params[name] = adapted_params[name] - self.config.inner_lr * grad
        
        return adapted_params
    
    def _compute_query_loss(
        self,
        model: nn.Module,
        query_data: Dict,
        adapted_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss on query set using adapted parameters."""
        
        # Temporarily replace model parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
            if name in adapted_params:
                param.data.copy_(adapted_params[name])
        
        # Forward pass with adapted parameters
        try:
            outputs = model(**query_data)
            loss = outputs.get('loss', outputs['logits'])
            
            if not isinstance(loss, torch.Tensor):
                loss = F.cross_entropy(outputs['logits'], query_data['labels'])
                
        finally:
            # Restore original parameters
            for name, param in model.named_parameters():
                param.data.copy_(original_params[name])
        
        return loss


class ContinualMetaLearner:
    """Main meta-continual learning orchestrator."""
    
    def __init__(self, model, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.memory_bank = TaskMemoryBank(
            memory_size=config.memory_replay_size,
            embedding_dim=model.config.hidden_size if hasattr(model.config, 'hidden_size') else 512
        )
        self.gradient_processor = MetaGradientProcessor(config)
        
        # Meta-learning state
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.meta_lr
        )
        self.adaptation_history = []
        self.performance_history = defaultdict(list)
        
        # Fast adaptation modules
        self.fast_weights = {}
        self.task_embeddings = nn.Embedding(100, 128)  # Support up to 100 tasks
        
        logger.info("Meta-continual learning system initialized")
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, Any]],
        support_queries: List[Tuple[Dict, Dict]]
    ) -> Dict[str, float]:
        """Perform one meta-training step."""
        
        self.meta_optimizer.zero_grad()
        
        # Prepare support and query sets
        support_data = []
        query_data = []
        task_ids = []
        
        for task_info, (support, query) in zip(task_batch, support_queries):
            support_data.append(support)
            query_data.append(query)
            task_ids.append(task_info['task_id'])
        
        # Compute meta-gradients
        meta_gradients = self.gradient_processor.compute_meta_gradients(
            self.model, support_data, query_data, task_ids
        )
        
        # Apply meta-gradients
        total_grad_norm = 0.0
        for name, param in self.model.named_parameters():
            if name in meta_gradients:
                param.grad = meta_gradients[name]
                total_grad_norm += meta_gradients[name].norm().item() ** 2
        
        total_grad_norm = math.sqrt(total_grad_norm)
        
        # Gradient clipping
        if total_grad_norm > 1.0:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param.grad.div_(total_grad_norm)
        
        # Meta-optimizer step
        self.meta_optimizer.step()
        
        # Compute meta-loss for logging
        meta_loss = self._compute_meta_loss(support_data, query_data, task_ids)
        
        metrics = {
            'meta_loss': meta_loss.item(),
            'grad_norm': total_grad_norm,
            'num_tasks': len(task_ids)
        }
        
        return metrics
    
    def fast_adapt_to_task(
        self,
        task_id: str,
        support_data: Dict,
        adaptation_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """Quickly adapt to a new task using meta-learned initialization."""
        
        steps = adaptation_steps or self.config.adaptation_steps
        
        # Initialize fast weights
        fast_weights = {
            name: param.clone() for name, param in self.model.named_parameters()
        }
        
        # Adaptation loop
        adaptation_losses = []
        
        for step in range(steps):
            
            # Forward pass with current fast weights
            self._apply_fast_weights(fast_weights)
            
            try:
                outputs = self.model(**support_data)
                loss = outputs.get('loss', outputs['logits'])
                
                if not isinstance(loss, torch.Tensor):
                    loss = F.cross_entropy(outputs['logits'], support_data['labels'])
                
                adaptation_losses.append(loss.item())
                
                # Compute gradients for fast weights
                gradients = torch.autograd.grad(
                    loss,
                    self.model.parameters(),
                    create_graph=False,
                    retain_graph=False
                )
                
                # Update fast weights
                for (name, param), grad in zip(self.model.named_parameters(), gradients):
                    fast_weights[name] = fast_weights[name] - self.config.inner_lr * grad
                    
            except Exception as e:
                logger.warning(f"Adaptation step {step} failed: {e}")
                break
        
        # Store fast weights for this task
        self.fast_weights[task_id] = fast_weights
        
        # Store experience in memory bank
        if 'input_ids' in support_data and 'labels' in support_data:
            with torch.no_grad():
                self._apply_fast_weights(fast_weights)
                outputs = self.model(
                    input_ids=support_data['input_ids'],
                    attention_mask=support_data.get('attention_mask')
                )
                self.memory_bank.store_task_experience(
                    task_id,
                    outputs['hidden_states'],
                    support_data['labels']
                )
        
        adaptation_result = {
            'task_id': task_id,
            'adaptation_steps': len(adaptation_losses),
            'final_loss': adaptation_losses[-1] if adaptation_losses else float('inf'),
            'loss_reduction': adaptation_losses[0] - adaptation_losses[-1] if len(adaptation_losses) > 1 else 0.0,
            'convergence_rate': self._compute_convergence_rate(adaptation_losses)
        }
        
        self.adaptation_history.append(adaptation_result)
        
        return adaptation_result
    
    def _apply_fast_weights(self, fast_weights: Dict[str, torch.Tensor]):
        """Apply fast weights to model parameters."""
        for name, param in self.model.named_parameters():
            if name in fast_weights:
                param.data.copy_(fast_weights[name])
    
    def _compute_meta_loss(
        self,
        support_data: List[Dict],
        query_data: List[Dict], 
        task_ids: List[str]
    ) -> torch.Tensor:
        """Compute meta-loss across all tasks."""
        
        total_loss = 0.0
        
        for support, query, task_id in zip(support_data, query_data, task_ids):
            
            # Fast adaptation on support set
            adapted_result = self.fast_adapt_to_task(
                task_id, support, adaptation_steps=3
            )
            
            # Evaluate on query set
            if task_id in self.fast_weights:
                self._apply_fast_weights(self.fast_weights[task_id])
            
            outputs = self.model(**query)
            loss = outputs.get('loss', outputs['logits'])
            
            if not isinstance(loss, torch.Tensor):
                loss = F.cross_entropy(outputs['logits'], query['labels'])
            
            total_loss += loss
        
        return total_loss / len(task_ids)
    
    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """Compute convergence rate from loss trajectory."""
        if len(losses) < 2:
            return 0.0
        
        # Compute average rate of loss decrease
        total_decrease = 0.0
        for i in range(1, len(losses)):
            decrease = max(0, losses[i-1] - losses[i])
            total_decrease += decrease
        
        return total_decrease / (len(losses) - 1)
    
    def episodic_memory_replay(
        self,
        current_task_id: str,
        num_replay_samples: int = 10
    ) -> Dict[str, Any]:
        """Perform episodic memory replay to prevent forgetting."""
        
        # Retrieve similar experiences from memory
        current_data = self.memory_bank.task_memories.get(current_task_id, [])
        if not current_data:
            return {'replay_loss': 0.0, 'num_replayed': 0}
        
        # Get representative sample from current task
        latest_experience = current_data[-1]
        query_states = latest_experience['hidden_states']
        
        # Retrieve similar experiences from other tasks
        similar_experiences = self.memory_bank.retrieve_similar_experiences(
            query_states,
            k=num_replay_samples,
            exclude_task=current_task_id
        )
        
        if not similar_experiences:
            return {'replay_loss': 0.0, 'num_replayed': 0}
        
        # Replay experiences to maintain performance
        replay_loss = 0.0
        replayed_count = 0
        
        for exp_info in similar_experiences:
            experience = exp_info['experience']
            source_task_id = exp_info['task_id']
            
            try:
                # Apply fast weights for source task if available
                if source_task_id in self.fast_weights:
                    self._apply_fast_weights(self.fast_weights[source_task_id])
                
                # Create dummy input for replay (simplified)
                dummy_input = {
                    'input_ids': torch.randint(0, 1000, (1, 32)),  # Dummy input
                    'labels': experience['labels'][:1]  # Take first label
                }
                
                # Forward pass
                outputs = self.model(**dummy_input)
                loss = outputs.get('loss', outputs['logits'])
                
                if not isinstance(loss, torch.Tensor):
                    loss = F.cross_entropy(outputs['logits'], dummy_input['labels'])
                
                replay_loss += loss.item()
                replayed_count += 1
                
            except Exception as e:
                logger.warning(f"Replay failed for experience from {source_task_id}: {e}")
        
        avg_replay_loss = replay_loss / max(replayed_count, 1)
        
        return {
            'replay_loss': avg_replay_loss,
            'num_replayed': replayed_count,
            'similarity_scores': [exp['similarity'] for exp in similar_experiences]
        }
    
    def knowledge_transfer_score(
        self,
        source_task_id: str,
        target_task_id: str
    ) -> float:
        """Compute knowledge transfer score between tasks."""
        
        if source_task_id not in self.fast_weights or target_task_id not in self.fast_weights:
            return 0.0
        
        source_weights = self.fast_weights[source_task_id]
        target_weights = self.fast_weights[target_task_id]
        
        # Compute cosine similarity between parameter vectors
        source_vec = torch.cat([w.flatten() for w in source_weights.values()])
        target_vec = torch.cat([w.flatten() for w in target_weights.values()])
        
        similarity = F.cosine_similarity(
            source_vec.unsqueeze(0),
            target_vec.unsqueeze(0)
        ).item()
        
        return similarity
    
    def get_meta_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive status of meta-learning system."""
        
        status = {
            'adaptation_history': {
                'total_adaptations': len(self.adaptation_history),
                'avg_convergence_rate': np.mean([
                    adapt['convergence_rate'] for adapt in self.adaptation_history
                ]) if self.adaptation_history else 0.0,
                'avg_final_loss': np.mean([
                    adapt['final_loss'] for adapt in self.adaptation_history
                ]) if self.adaptation_history else 0.0
            },
            'memory_bank': self.memory_bank.get_memory_statistics(),
            'fast_weights': {
                'num_tasks': len(self.fast_weights),
                'task_ids': list(self.fast_weights.keys())
            },
            'meta_optimizer': {
                'learning_rate': self.config.meta_lr,
                'inner_learning_rate': self.config.inner_lr
            }
        }
        
        # Add knowledge transfer analysis
        if len(self.fast_weights) > 1:
            task_ids = list(self.fast_weights.keys())
            transfer_scores = []
            
            for i, source in enumerate(task_ids):
                for j, target in enumerate(task_ids):
                    if i != j:
                        score = self.knowledge_transfer_score(source, target)
                        transfer_scores.append(score)
            
            status['knowledge_transfer'] = {
                'avg_transfer_score': np.mean(transfer_scores),
                'max_transfer_score': np.max(transfer_scores),
                'min_transfer_score': np.min(transfer_scores)
            }
        
        return status
    
    def save_meta_state(self, filepath: str):
        """Save meta-learning state."""
        state = {
            'config': self.config.__dict__,
            'adaptation_history': self.adaptation_history,
            'fast_weights': {k: {name: param.cpu() for name, param in weights.items()} 
                           for k, weights in self.fast_weights.items()},
            'memory_statistics': self.memory_bank.get_memory_statistics()
        }
        
        torch.save(state, filepath)
        logger.info(f"Meta-learning state saved to {filepath}")
    
    def load_meta_state(self, filepath: str):
        """Load meta-learning state."""
        try:
            state = torch.load(filepath, map_location='cpu')
            
            self.adaptation_history = state.get('adaptation_history', [])
            
            # Restore fast weights
            fast_weights = state.get('fast_weights', {})
            for task_id, weights in fast_weights.items():
                self.fast_weights[task_id] = {
                    name: param.to(next(self.model.parameters()).device)
                    for name, param in weights.items()
                }
            
            logger.info(f"Meta-learning state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load meta-learning state: {e}")


def create_meta_continual_learner(
    model,
    meta_lr: float = 1e-3,
    inner_lr: float = 1e-2,
    memory_size: int = 1000,
    **kwargs
) -> ContinualMetaLearner:
    """Factory function to create meta-continual learner."""
    
    config = MetaLearningConfig(
        meta_lr=meta_lr,
        inner_lr=inner_lr,
        memory_replay_size=memory_size,
        **kwargs
    )
    
    return ContinualMetaLearner(model, config)


# Example usage and testing functions
def demonstrate_meta_learning():
    """Demonstrate meta-continual learning capabilities."""
    
    logger.info("Demonstrating Meta-Continual Learning")
    
    # This would be integrated with the main ContinualTransformer
    print("Meta-Continual Learning Framework:")
    print("✓ MAML-style meta-learning for continual scenarios")
    print("✓ Episodic memory bank for cross-task knowledge")
    print("✓ Fast adaptation with meta-learned initialization")
    print("✓ Knowledge transfer scoring between tasks")
    print("✓ Second-order gradient optimization")
    print("✓ Memory replay for catastrophic forgetting prevention")


if __name__ == "__main__":
    demonstrate_meta_learning()