"""Advanced knowledge transfer and meta-learning optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from dataclasses import dataclass
import copy
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeTransferMetrics:
    """Metrics for knowledge transfer performance."""
    transfer_accuracy: float
    forgetting_rate: float
    knowledge_retention: float
    cross_task_similarity: float
    adaptation_speed: float


class KnowledgeTransferOptimizer:
    """Advanced knowledge transfer optimization for continual learning."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Knowledge bases
        self.task_knowledge_bases = {}
        self.shared_knowledge_base = None
        
        # Transfer strategies
        self.transfer_strategies = {
            'gradient_based': self._gradient_based_transfer,
            'feature_based': self._feature_based_transfer,
            'parameter_based': self._parameter_based_transfer,
            'attention_based': self._attention_based_transfer
        }
        
        # Meta-learning components
        self.meta_learner = None
        self.task_embeddings = {}
        
        # Performance tracking
        self.transfer_history = []
        
    def extract_task_knowledge(self, task_id: str, dataloader) -> Dict[str, Any]:
        """Extract and store knowledge from a learned task."""
        
        knowledge_base = {
            'task_id': task_id,
            'feature_statistics': self._compute_feature_statistics(dataloader),
            'gradient_patterns': self._analyze_gradient_patterns(dataloader),
            'attention_patterns': self._extract_attention_patterns(dataloader),
            'parameter_importance': self._compute_parameter_importance(dataloader),
            'task_embedding': self._compute_task_embedding(dataloader)
        }
        
        self.task_knowledge_bases[task_id] = knowledge_base
        self._update_shared_knowledge_base(knowledge_base)
        
        logger.info(f"Extracted knowledge for task: {task_id}")
        return knowledge_base
    
    def _compute_feature_statistics(self, dataloader) -> Dict[str, torch.Tensor]:
        """Compute statistical properties of features for the task."""
        
        self.model.eval()
        feature_stats = {
            'mean': None,
            'std': None,
            'min_vals': None,
            'max_vals': None,
            'covariance': None
        }
        
        all_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items() if k != 'labels'}
                    outputs = self.model(**inputs)
                else:
                    # Handle non-dict batch format
                    inputs = batch[0].to(self.config.device)
                    outputs = self.model(inputs)
                
                if 'hidden_states' in outputs:
                    features = outputs['hidden_states']
                elif 'pooled_output' in outputs:
                    features = outputs['pooled_output']
                else:
                    continue
                
                # Flatten features for statistics
                batch_size = features.size(0)
                features_flat = features.view(batch_size, -1)
                all_features.append(features_flat)
        
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            
            feature_stats['mean'] = torch.mean(all_features, dim=0)
            feature_stats['std'] = torch.std(all_features, dim=0)
            feature_stats['min_vals'] = torch.min(all_features, dim=0)[0]
            feature_stats['max_vals'] = torch.max(all_features, dim=0)[0]
            
            # Compute covariance matrix (for smaller feature sets)
            if all_features.size(1) <= 1000:
                feature_stats['covariance'] = torch.cov(all_features.T)
        
        return feature_stats
    
    def _analyze_gradient_patterns(self, dataloader) -> Dict[str, torch.Tensor]:
        """Analyze gradient patterns specific to the task."""
        
        self.model.train()
        gradient_patterns = {
            'mean_gradients': {},
            'gradient_variance': {},
            'gradient_directions': {}
        }
        
        # Collect gradients over multiple batches
        accumulated_gradients = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 20:  # Limit to 20 batches for efficiency
                break
            
            self.model.zero_grad()
            
            # Forward pass
            if isinstance(batch, dict):
                inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                         for k, v in batch.items()}
                outputs = self.model(**inputs)
            else:
                inputs = batch[0].to(self.config.device)
                labels = batch[1].to(self.config.device)
                outputs = self.model(inputs, labels=labels)
            
            if 'loss' in outputs:
                loss = outputs['loss']
                loss.backward()
                
                # Collect gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None and 'adapter' in name:
                        accumulated_gradients[name].append(param.grad.clone())
        
        # Compute statistics
        for name, grads in accumulated_gradients.items():
            if grads:
                stacked_grads = torch.stack(grads)
                gradient_patterns['mean_gradients'][name] = torch.mean(stacked_grads, dim=0)
                gradient_patterns['gradient_variance'][name] = torch.var(stacked_grads, dim=0)
                
                # Compute dominant gradient direction
                grad_flat = stacked_grads.view(len(grads), -1)
                U, S, V = torch.svd(grad_flat)
                gradient_patterns['gradient_directions'][name] = V[:, 0]  # First principal component
        
        return gradient_patterns
    
    def _extract_attention_patterns(self, dataloader) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from the model."""
        
        # This would require access to attention weights
        # For now, return placeholder
        return {
            'attention_entropy': torch.zeros(1),
            'attention_sparsity': torch.zeros(1),
            'head_importance': torch.zeros(12)  # Assuming 12 attention heads
        }
    
    def _compute_parameter_importance(self, dataloader) -> Dict[str, torch.Tensor]:
        """Compute importance scores for model parameters."""
        
        # Fisher Information Matrix approximation
        self.model.eval()
        importance_scores = {}
        
        # Initialize importance scores
        for name, param in self.model.named_parameters():
            if 'adapter' in name and param.requires_grad:
                importance_scores[name] = torch.zeros_like(param)
        
        # Accumulate importance over data
        num_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50:  # Limit for efficiency
                break
            
            self.model.zero_grad()
            
            # Forward pass
            if isinstance(batch, dict):
                inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                         for k, v in batch.items()}
                outputs = self.model(**inputs)
            else:
                inputs = batch[0].to(self.config.device)
                labels = batch[1].to(self.config.device)
                outputs = self.model(inputs, labels=labels)
            
            if 'loss' in outputs:
                loss = outputs['loss']
                loss.backward()
                
                # Accumulate squared gradients (Fisher Information approximation)
                for name, param in self.model.named_parameters():
                    if name in importance_scores and param.grad is not None:
                        importance_scores[name] += param.grad.data ** 2
                
                num_samples += inputs.get('input_ids', inputs).size(0)
        
        # Normalize by number of samples
        for name in importance_scores:
            importance_scores[name] /= max(num_samples, 1)
        
        return importance_scores
    
    def _compute_task_embedding(self, dataloader) -> torch.Tensor:
        """Compute a compact embedding representing the task."""
        
        # Collect features from the task
        self.model.eval()
        all_features = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 10:  # Limit for efficiency
                    break
                
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items() if k != 'labels'}
                    outputs = self.model(**inputs)
                else:
                    inputs = batch[0].to(self.config.device)
                    outputs = self.model(inputs)
                
                if 'pooled_output' in outputs:
                    features = outputs['pooled_output']
                    all_features.append(features.mean(dim=0))
        
        if all_features:
            # Compute mean feature representation
            task_embedding = torch.stack(all_features).mean(dim=0)
        else:
            # Fallback to random embedding
            task_embedding = torch.randn(768)  # Default size
        
        return task_embedding
    
    def _update_shared_knowledge_base(self, task_knowledge: Dict[str, Any]):
        """Update the shared knowledge base with new task knowledge."""
        
        if self.shared_knowledge_base is None:
            self.shared_knowledge_base = {
                'task_embeddings': [],
                'common_patterns': {},
                'transfer_matrix': torch.zeros(0, 0)
            }
        
        # Add task embedding to shared knowledge
        task_embedding = task_knowledge['task_embedding']
        self.shared_knowledge_base['task_embeddings'].append(task_embedding)
        
        # Update transfer matrix
        num_tasks = len(self.shared_knowledge_base['task_embeddings'])
        if num_tasks > 1:
            self._update_transfer_matrix()
    
    def _update_transfer_matrix(self):
        """Update the task-to-task transfer similarity matrix."""
        
        embeddings = self.shared_knowledge_base['task_embeddings']
        num_tasks = len(embeddings)
        
        transfer_matrix = torch.zeros(num_tasks, num_tasks)
        
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    # Compute cosine similarity
                    sim = F.cosine_similarity(embeddings[i], embeddings[j], dim=0)
                    transfer_matrix[i, j] = sim
        
        self.shared_knowledge_base['transfer_matrix'] = transfer_matrix
    
    def find_best_source_tasks(self, target_task_id: str, num_sources: int = 3) -> List[str]:
        """Find the best source tasks for transferring knowledge to the target task."""
        
        if target_task_id not in self.task_knowledge_bases:
            logger.warning(f"Target task {target_task_id} not found in knowledge bases")
            return list(self.task_knowledge_bases.keys())[:num_sources]
        
        target_embedding = self.task_knowledge_bases[target_task_id]['task_embedding']
        
        # Compute similarities with all other tasks
        similarities = {}
        for source_task_id, source_knowledge in self.task_knowledge_bases.items():
            if source_task_id != target_task_id:
                source_embedding = source_knowledge['task_embedding']
                similarity = F.cosine_similarity(target_embedding, source_embedding, dim=0)
                similarities[source_task_id] = similarity.item()
        
        # Sort by similarity and return top sources
        sorted_sources = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_sources = [task_id for task_id, _ in sorted_sources[:num_sources]]
        
        logger.info(f"Best source tasks for {target_task_id}: {best_sources}")
        return best_sources
    
    def transfer_knowledge(
        self, 
        source_task_ids: List[str], 
        target_task_id: str,
        transfer_strategy: str = 'gradient_based'
    ) -> Dict[str, Any]:
        """Transfer knowledge from source tasks to target task."""
        
        if transfer_strategy not in self.transfer_strategies:
            raise ValueError(f"Unknown transfer strategy: {transfer_strategy}")
        
        transfer_fn = self.transfer_strategies[transfer_strategy]
        
        # Perform knowledge transfer
        transfer_result = transfer_fn(source_task_ids, target_task_id)
        
        # Record transfer attempt
        self.transfer_history.append({
            'source_tasks': source_task_ids,
            'target_task': target_task_id,
            'strategy': transfer_strategy,
            'timestamp': torch.tensor(time.time()),
            'result': transfer_result
        })
        
        logger.info(
            f"Transferred knowledge from {source_task_ids} to {target_task_id} "
            f"using {transfer_strategy} strategy"
        )
        
        return transfer_result
    
    def _gradient_based_transfer(self, source_task_ids: List[str], target_task_id: str) -> Dict[str, Any]:
        """Transfer knowledge using gradient-based alignment."""
        
        # Get gradient patterns from source tasks
        source_gradients = []
        for source_id in source_task_ids:
            if source_id in self.task_knowledge_bases:
                grad_patterns = self.task_knowledge_bases[source_id]['gradient_patterns']
                source_gradients.append(grad_patterns)
        
        if not source_gradients:
            return {'success': False, 'reason': 'No source gradients available'}
        
        # Compute average gradient patterns
        avg_gradients = {}
        for param_name in source_gradients[0]['mean_gradients']:
            grads = [sg['mean_gradients'][param_name] for sg in source_gradients 
                    if param_name in sg['mean_gradients']]
            if grads:
                avg_gradients[param_name] = torch.stack(grads).mean(dim=0)
        
        # Apply gradient-based initialization to target task adapter
        target_adapter = self.model.adapters.get(target_task_id)
        if target_adapter is not None:
            self._apply_gradient_initialization(target_adapter, avg_gradients)
        
        return {
            'success': True,
            'transferred_parameters': len(avg_gradients),
            'method': 'gradient_based'
        }
    
    def _feature_based_transfer(self, source_task_ids: List[str], target_task_id: str) -> Dict[str, Any]:
        """Transfer knowledge using feature distribution alignment."""
        
        # Get feature statistics from source tasks
        source_stats = []
        for source_id in source_task_ids:
            if source_id in self.task_knowledge_bases:
                stats = self.task_knowledge_bases[source_id]['feature_statistics']
                source_stats.append(stats)
        
        if not source_stats:
            return {'success': False, 'reason': 'No source statistics available'}
        
        # Compute average feature statistics
        avg_stats = {}
        for key in ['mean', 'std']:
            if all(key in stats for stats in source_stats):
                values = [stats[key] for stats in source_stats if stats[key] is not None]
                if values:
                    avg_stats[key] = torch.stack(values).mean(dim=0)
        
        # Apply feature normalization to target adapter
        target_adapter = self.model.adapters.get(target_task_id)
        if target_adapter is not None and avg_stats:
            self._apply_feature_normalization(target_adapter, avg_stats)
        
        return {
            'success': True,
            'transferred_statistics': len(avg_stats),
            'method': 'feature_based'
        }
    
    def _parameter_based_transfer(self, source_task_ids: List[str], target_task_id: str) -> Dict[str, Any]:
        """Transfer knowledge using parameter importance weighting."""
        
        # Get parameter importance from source tasks
        source_importance = []
        for source_id in source_task_ids:
            if source_id in self.task_knowledge_bases:
                importance = self.task_knowledge_bases[source_id]['parameter_importance']
                source_importance.append(importance)
        
        if not source_importance:
            return {'success': False, 'reason': 'No source importance available'}
        
        # Compute weighted average importance
        avg_importance = {}
        for param_name in source_importance[0]:
            importances = [si[param_name] for si in source_importance if param_name in si]
            if importances:
                avg_importance[param_name] = torch.stack(importances).mean(dim=0)
        
        # Initialize target adapter parameters based on importance
        target_adapter = self.model.adapters.get(target_task_id)
        if target_adapter is not None:
            self._apply_importance_initialization(target_adapter, avg_importance)
        
        return {
            'success': True,
            'transferred_importance': len(avg_importance),
            'method': 'parameter_based'
        }
    
    def _attention_based_transfer(self, source_task_ids: List[str], target_task_id: str) -> Dict[str, Any]:
        """Transfer knowledge using attention pattern alignment."""
        
        # Get attention patterns from source tasks
        source_attention = []
        for source_id in source_task_ids:
            if source_id in self.task_knowledge_bases:
                attention = self.task_knowledge_bases[source_id]['attention_patterns']
                source_attention.append(attention)
        
        if not source_attention:
            return {'success': False, 'reason': 'No source attention patterns available'}
        
        # This is a simplified implementation
        # In practice, would involve more sophisticated attention transfer
        
        return {
            'success': True,
            'method': 'attention_based'
        }
    
    def _apply_gradient_initialization(self, adapter: nn.Module, gradient_patterns: Dict[str, torch.Tensor]):
        """Initialize adapter parameters based on gradient patterns."""
        
        for name, param in adapter.named_parameters():
            if name in gradient_patterns:
                with torch.no_grad():
                    # Scale initialization based on gradient magnitude
                    grad_magnitude = torch.norm(gradient_patterns[name])
                    scale_factor = min(grad_magnitude.item(), 1.0)
                    
                    # Apply scaled initialization
                    nn.init.normal_(param, mean=0.0, std=0.02 * scale_factor)
    
    def _apply_feature_normalization(self, adapter: nn.Module, feature_stats: Dict[str, torch.Tensor]):
        """Apply feature normalization to adapter based on source statistics."""
        
        # Add batch normalization layers if not present
        for module in adapter.modules():
            if isinstance(module, nn.Linear):
                # Adjust weights based on feature statistics
                with torch.no_grad():
                    if 'std' in feature_stats and feature_stats['std'] is not None:
                        # Scale weights by feature standard deviation
                        std_mean = feature_stats['std'].mean()
                        if std_mean > 0:
                            module.weight.data *= (1.0 / std_mean)
    
    def _apply_importance_initialization(self, adapter: nn.Module, importance_scores: Dict[str, torch.Tensor]):
        """Initialize adapter based on parameter importance scores."""
        
        for name, param in adapter.named_parameters():
            # Find matching importance pattern
            matching_importance = None
            for imp_name, imp_scores in importance_scores.items():
                if name.split('.')[-1] == imp_name.split('.')[-1]:  # Match layer names
                    if param.shape == imp_scores.shape:
                        matching_importance = imp_scores
                        break
            
            if matching_importance is not None:
                with torch.no_grad():
                    # Initialize with importance-weighted random values
                    importance_weights = torch.sqrt(matching_importance + 1e-8)
                    param.data.normal_(0, 0.02)
                    param.data *= importance_weights


class CrossTaskTransfer:
    """Cross-task knowledge transfer mechanism."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.task_similarities = {}
        
    def compute_task_similarity(self, task1_id: str, task2_id: str, dataloader1, dataloader2) -> float:
        """Compute similarity between two tasks."""
        
        # Extract features from both tasks
        features1 = self._extract_task_features(task1_id, dataloader1)
        features2 = self._extract_task_features(task2_id, dataloader2)
        
        if features1 is None or features2 is None:
            return 0.0
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(features1, features2, dim=0)
        
        # Store similarity for future use
        self.task_similarities[(task1_id, task2_id)] = similarity.item()
        self.task_similarities[(task2_id, task1_id)] = similarity.item()
        
        return similarity.item()
    
    def _extract_task_features(self, task_id: str, dataloader) -> Optional[torch.Tensor]:
        """Extract representative features for a task."""
        
        self.model.eval()
        self.model.set_current_task(task_id)
        
        all_features = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 20:  # Limit batches
                    break
                
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items() if k != 'labels'}
                    outputs = self.model(**inputs)
                else:
                    inputs = batch[0].to(self.config.device)
                    outputs = self.model(inputs)
                
                if 'pooled_output' in outputs:
                    all_features.append(outputs['pooled_output'].mean(dim=0))
        
        if all_features:
            return torch.stack(all_features).mean(dim=0)
        
        return None
    
    def adaptive_transfer_weight(self, source_task_id: str, target_task_id: str) -> float:
        """Compute adaptive transfer weight based on task similarity."""
        
        similarity = self.task_similarities.get((source_task_id, target_task_id), 0.0)
        
        # Sigmoid-based adaptive weighting
        transfer_weight = torch.sigmoid(torch.tensor(similarity * 5.0 - 2.5)).item()
        
        return transfer_weight


class MetaLearningOptimizer:
    """Meta-learning optimization for few-shot task adaptation."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Meta-learning components
        self.meta_optimizer = None
        self.task_distribution = {}
        
    def initialize_meta_learning(self):
        """Initialize meta-learning components."""
        
        # Create meta-optimizer for adapter parameters
        adapter_params = []
        for adapter in self.model.adapters.values():
            adapter_params.extend(adapter.parameters())
        
        self.meta_optimizer = torch.optim.Adam(adapter_params, lr=1e-3)
    
    def meta_train_step(self, support_data, query_data, task_id: str):
        """Perform one meta-training step."""
        
        if self.meta_optimizer is None:
            self.initialize_meta_learning()
        
        # Support phase: adapt to task
        self.model.set_current_task(task_id)
        support_loss = self._compute_support_loss(support_data, task_id)
        
        # Compute gradients
        support_loss.backward(create_graph=True)
        
        # Query phase: test adaptation
        query_loss = self._compute_query_loss(query_data, task_id)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        
        return {
            'support_loss': support_loss.item(),
            'query_loss': query_loss.item()
        }
    
    def _compute_support_loss(self, support_data, task_id: str) -> torch.Tensor:
        """Compute loss on support set."""
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in support_data:
            if isinstance(batch, dict):
                inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                         for k, v in batch.items()}
                outputs = self.model(**inputs)
            else:
                inputs = batch[0].to(self.config.device)
                labels = batch[1].to(self.config.device)
                outputs = self.model(inputs, labels=labels, task_id=task_id)
            
            if 'loss' in outputs:
                total_loss += outputs['loss']
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _compute_query_loss(self, query_data, task_id: str) -> torch.Tensor:
        """Compute loss on query set."""
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in query_data:
            if isinstance(batch, dict):
                inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                         for k, v in batch.items()}
                outputs = self.model(**inputs)
            else:
                inputs = batch[0].to(self.config.device)
                labels = batch[1].to(self.config.device)
                outputs = self.model(inputs, labels=labels, task_id=task_id)
            
            if 'loss' in outputs:
                total_loss += outputs['loss']
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def few_shot_adapt(self, task_id: str, few_shot_data, num_steps: int = 5):
        """Perform few-shot adaptation to a new task."""
        
        self.model.set_current_task(task_id)
        
        # Fast adaptation using gradient descent
        for step in range(num_steps):
            total_loss = 0.0
            
            for batch in few_shot_data:
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items()}
                    outputs = self.model(**inputs)
                else:
                    inputs = batch[0].to(self.config.device)
                    labels = batch[1].to(self.config.device)
                    outputs = self.model(inputs, labels=labels, task_id=task_id)
                
                if 'loss' in outputs:
                    total_loss += outputs['loss']
            
            # Update only adapter parameters
            adapter = self.model.adapters[task_id]
            adapter_optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)
            
            adapter_optimizer.zero_grad()
            total_loss.backward()
            adapter_optimizer.step()
        
        logger.info(f"Few-shot adaptation completed for task {task_id}")
        
        return {'adaptation_steps': num_steps, 'final_loss': total_loss.item()}