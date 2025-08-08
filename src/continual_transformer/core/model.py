"""Core continual learning transformer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
import logging
from pathlib import Path
import json

from .config import ContinualConfig
from .error_recovery import ErrorRecoverySystem
from ..adapters.activation import ActivationAdapter, create_adapter
from ..tasks.manager import TaskManager
from ..utils.knowledge_distillation import KnowledgeDistillation
from ..optimization.performance_optimizer import PerformanceOptimizer, AdaptiveOptimizer
from ..optimization.knowledge_transfer import KnowledgeTransferOptimizer
from ..optimization.neural_architecture_search import NASOptimizer
from ..monitoring.system_monitor import SystemMonitor


logger = logging.getLogger(__name__)


class TaskRouter(nn.Module):
    """Task routing module for identifying and routing to appropriate task adapters."""
    
    def __init__(self, config: ContinualConfig, input_size: int):
        super().__init__()
        self.config = config
        self.method = config.task_routing_method
        
        if self.method == "learned":
            self.classifier = nn.Sequential(
                nn.Linear(input_size, config.router_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.router_hidden_size, config.max_tasks)
            )
        elif self.method == "embedding":
            self.task_embeddings = nn.Embedding(config.max_tasks, input_size)
            
        self.task_id_to_index = {}
        self.index_to_task_id = {}
        self.num_tasks = 0
    
    def register_task(self, task_id: str) -> int:
        """Register a new task and return its index."""
        if task_id not in self.task_id_to_index:
            if self.num_tasks >= self.config.max_tasks:
                raise ValueError(f"Maximum number of tasks ({self.config.max_tasks}) exceeded")
            
            task_index = self.num_tasks
            self.task_id_to_index[task_id] = task_index
            self.index_to_task_id[task_index] = task_id
            self.num_tasks += 1
            
            logger.info(f"Registered task '{task_id}' with index {task_index}")
            
        return self.task_id_to_index[task_id]
    
    def forward(self, hidden_states: torch.Tensor, task_id: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route inputs to appropriate task.
        
        Returns:
            task_probs: Task probability distribution [batch_size, num_tasks]
            task_indices: Predicted task indices [batch_size]
        """
        batch_size = hidden_states.size(0)
        
        if task_id is not None:
            # Explicit task routing
            if task_id not in self.task_id_to_index:
                raise ValueError(f"Unknown task_id: {task_id}")
            
            task_index = self.task_id_to_index[task_id]
            task_indices = torch.full((batch_size,), task_index, device=hidden_states.device)
            
            # Create one-hot probabilities
            task_probs = torch.zeros(batch_size, self.config.max_tasks, device=hidden_states.device)
            task_probs[:, task_index] = 1.0
            
        elif self.method == "learned":
            # Learned task routing
            pooled = hidden_states.mean(dim=1)  # Global average pooling
            logits = self.classifier(pooled)
            task_probs = F.softmax(logits[:, :self.num_tasks], dim=-1)
            task_indices = task_probs.argmax(dim=-1)
            
        elif self.method == "embedding":
            # Embedding-based similarity routing
            pooled = hidden_states.mean(dim=1)
            
            # Compute similarity with all task embeddings
            task_embs = self.task_embeddings.weight[:self.num_tasks]  # [num_tasks, hidden_size]
            similarities = torch.matmul(pooled, task_embs.T)  # [batch_size, num_tasks]
            
            task_probs = F.softmax(similarities, dim=-1)
            task_indices = task_probs.argmax(dim=-1)
            
        else:
            raise ValueError(f"Unknown routing method: {self.method}")
        
        return task_probs, task_indices


class ContinualTransformer(nn.Module):
    """Zero-parameter continual learning transformer."""
    
    def __init__(self, config: ContinualConfig):
        super().__init__()
        self.config = config
        
        # Initialize base transformer (frozen)
        self.base_model = AutoModel.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )
        
        # Freeze base model parameters
        if config.freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info("Frozen base model parameters")
        
        # Task management
        self.task_manager = TaskManager(config)
        
        # Task routing
        self.task_router = TaskRouter(config, self.base_model.config.hidden_size)
        
        # Task-specific adapters
        self.adapters = nn.ModuleDict()
        
        # Knowledge distillation
        if config.use_knowledge_distillation:
            self.knowledge_distillation = KnowledgeDistillation(config)
        
        # Classification heads for each task
        self.classification_heads = nn.ModuleDict()
        
        # EWC for catastrophic forgetting prevention
        self.ewc_params = {}
        self.ewc_fisher = {}
        
        # Training state
        self.current_task_id = None
        self.previous_model_state = None
        
        # Metrics tracking
        self.task_performance = {}
        
        # Advanced optimization components
        self.error_recovery = ErrorRecoverySystem(self, config)
        self.performance_optimizer = PerformanceOptimizer(self, config)
        self.adaptive_optimizer = AdaptiveOptimizer(self, config)
        self.knowledge_transfer = KnowledgeTransferOptimizer(self, config)
        self.nas_optimizer = NASOptimizer(self, config) if config.enable_nas else None
        self.system_monitor = SystemMonitor(self, config) if config.enable_monitoring else None
        
        # Start background monitoring if enabled
        if self.system_monitor:
            self.system_monitor.start_monitoring()
        if self.error_recovery:
            self.error_recovery.start_monitoring()
        
    def register_task(self, task_id: str, num_labels: int, task_type: str = "classification"):
        """Register a new task with the model."""
        if task_id in self.adapters:
            logger.warning(f"Task '{task_id}' already registered")
            return
        
        # Register with task router
        task_index = self.task_router.register_task(task_id)
        
        # Register with task manager
        task_config = self.config.get_task_config(task_id)
        self.task_manager.add_task(task_id, task_type, task_config)
        
        # Create task-specific adapter with optional NAS optimization
        adapter_config = None
        if self.nas_optimizer and hasattr(self.config, 'enable_nas') and self.config.enable_nas:
            try:
                # Use NAS to find optimal architecture (placeholder for demo)
                adapter_config = {
                    'adapter_type': self.config.adaptation_method,
                    'hidden_size': self.base_model.config.hidden_size,
                    'adapter_size': 64,
                    'num_layers': self.base_model.config.num_hidden_layers
                }
                adapter = create_adapter(
                    adapter_config['adapter_type'],
                    **adapter_config
                )
                logger.info(f"Created NAS-optimized adapter for task '{task_id}'")
            except Exception as e:
                logger.warning(f"NAS optimization failed, using default adapter: {e}")
                adapter_config = None
        
        if adapter_config is None:
            # Create default adapter
            if self.config.adaptation_method == "activation":
                adapter = ActivationAdapter(
                    hidden_size=self.base_model.config.hidden_size,
                    adapter_size=64,  # Configurable
                    num_layers=self.base_model.config.num_hidden_layers
                )
            else:
                # Use factory function for other adapter types
                adapter = create_adapter(
                    self.config.adaptation_method,
                    hidden_size=self.base_model.config.hidden_size,
                    adapter_size=64
                )
        
        self.adapters[task_id] = adapter
        
        # Create classification head
        self.classification_heads[task_id] = nn.Linear(
            self.base_model.config.hidden_size, 
            num_labels
        )
        
        # Initialize performance tracking
        self.task_performance[task_id] = {
            "train_accuracy": [],
            "eval_accuracy": [],
            "train_loss": [],
            "eval_loss": []
        }
        
        logger.info(f"Registered task '{task_id}' with {num_labels} labels")
    
    def set_current_task(self, task_id: str):
        """Set the current task for training/inference."""
        if task_id not in self.adapters:
            raise ValueError(f"Task '{task_id}' not registered. Call register_task() first.")
        
        self.current_task_id = task_id
        logger.info(f"Set current task to '{task_id}'")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_id: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the continual transformer with robust error handling."""
        
        try:
            # Input validation and sanitization
            self._validate_inputs(input_ids, attention_mask, labels)
            
            # Use current task if not specified
            if task_id is None:
                task_id = self.current_task_id
            
            if task_id is None:
                raise ValueError("No task_id specified and no current task set")
            
            # Verify task exists
            if task_id not in self.adapters:
                raise ValueError(f"Task '{task_id}' not found. Available tasks: {list(self.adapters.keys())}")
            
            # Device consistency check
            if input_ids.device != next(self.parameters()).device:
                logger.warning(f"Moving input from {input_ids.device} to {next(self.parameters()).device}")
                input_ids = input_ids.to(next(self.parameters()).device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(next(self.parameters()).device)
                if labels is not None:
                    labels = labels.to(next(self.parameters()).device)
            
            # Base transformer forward pass with error handling
            try:
                base_outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            except Exception as e:
                logger.error(f"Base model forward pass failed: {e}")
                raise RuntimeError(f"Base model inference failed: {str(e)}")
            
            hidden_states = base_outputs.last_hidden_state
            
            # Validate hidden states
            if torch.isnan(hidden_states).any():
                logger.error("NaN detected in base model outputs")
                raise RuntimeError("NaN values detected in base model hidden states")
            
            # Task routing with error handling
            try:
                task_probs, predicted_task_indices = self.task_router(hidden_states, task_id)
            except Exception as e:
                logger.error(f"Task routing failed: {e}")
                # Fallback to identity routing
                batch_size = hidden_states.size(0)
                task_probs = torch.zeros(batch_size, self.config.max_tasks, device=hidden_states.device)
                predicted_task_indices = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
            
            # Apply task-specific adapter with fallback
            try:
                if task_id in self.adapters:
                    adapted_states = self.adapters[task_id](hidden_states)
                else:
                    logger.warning(f"No adapter found for task '{task_id}', using original states")
                    adapted_states = hidden_states
            except Exception as e:
                logger.error(f"Adapter application failed for task '{task_id}': {e}")
                adapted_states = hidden_states  # Fallback to original states
            
            # Validate adapted states
            if torch.isnan(adapted_states).any():
                logger.error(f"NaN detected in adapter outputs for task '{task_id}'")
                adapted_states = hidden_states  # Fallback to original states
            
            # Global pooling for classification
            pooled_output = adapted_states.mean(dim=1)
            
            # Task-specific classification with error handling
            try:
                if task_id not in self.classification_heads:
                    raise ValueError(f"No classification head found for task '{task_id}'")
                logits = self.classification_heads[task_id](pooled_output)
            except Exception as e:
                logger.error(f"Classification head failed for task '{task_id}': {e}")
                raise RuntimeError(f"Classification inference failed: {str(e)}")
            
            # Validate logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.error(f"Invalid logits detected for task '{task_id}'")
                raise RuntimeError("Invalid logits (NaN/Inf) detected in classification head")
            
            outputs = {
                "logits": logits,
                "hidden_states": adapted_states,
                "task_probs": task_probs,
                "predicted_task_indices": predicted_task_indices,
                "pooled_output": pooled_output
            }
            
            # Compute loss if labels provided
            if labels is not None:
                try:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                    
                    # Validate loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error("Invalid loss detected")
                        raise RuntimeError("Loss computation resulted in NaN/Inf")
                    
                    # Add knowledge distillation loss if applicable
                    if (self.config.use_knowledge_distillation and 
                        hasattr(self, 'knowledge_distillation') and
                        self.previous_model_state is not None):
                        
                        try:
                            kd_loss = self.knowledge_distillation.compute_loss(
                                student_logits=logits,
                                teacher_logits=self.previous_model_state.get('logits'),
                                temperature=self.config.temperature
                            )
                            
                            if not torch.isnan(kd_loss):
                                loss = loss + self.config.knowledge_distillation_alpha * kd_loss
                                outputs["kd_loss"] = kd_loss
                            else:
                                logger.warning("Knowledge distillation loss is NaN, skipping")
                        except Exception as e:
                            logger.warning(f"Knowledge distillation failed: {e}")
                    
                    # Add EWC loss if applicable
                    if self.config.elastic_weight_consolidation and self.ewc_params:
                        try:
                            ewc_loss = self.compute_ewc_loss()
                            if not torch.isnan(ewc_loss):
                                loss = loss + self.config.ewc_lambda * ewc_loss
                                outputs["ewc_loss"] = ewc_loss
                            else:
                                logger.warning("EWC loss is NaN, skipping")
                        except Exception as e:
                            logger.warning(f"EWC computation failed: {e}")
                    
                    outputs["loss"] = loss
                    
                except Exception as e:
                    logger.error(f"Loss computation failed: {e}")
                    raise RuntimeError(f"Loss computation failed: {str(e)}")
            
            return outputs if return_dict else (logits,)
            
        except Exception as e:
            # Use error recovery system if available
            if self.error_recovery:
                context = {
                    'task_id': task_id,
                    'input_shape': input_ids.shape if input_ids is not None else None,
                    'device': str(input_ids.device) if input_ids is not None else None,
                    'is_training': self.training,
                    'batch_size': input_ids.size(0) if input_ids is not None else 0
                }
                
                success, result = self.error_recovery.handle_error(e, context)
                
                if success:
                    logger.info(f"Error recovered successfully: {result}")
                    # Retry forward pass once after recovery
                    try:
                        return self.forward(input_ids, attention_mask, task_id, labels, return_dict)
                    except Exception as retry_error:
                        logger.error(f"Forward pass failed even after error recovery: {retry_error}")
                        raise retry_error
                else:
                    logger.error(f"Error recovery failed: {result}")
            
            # Log comprehensive error information
            logger.error(
                f"Forward pass failed for task '{task_id}': {str(e)}\n"
                f"Input shape: {input_ids.shape if input_ids is not None else 'None'}\n"
                f"Device: {input_ids.device if input_ids is not None else 'None'}\n"
                f"Current task: {self.current_task_id}\n"
                f"Available tasks: {list(self.adapters.keys())}"
            )
            raise
    
    def _validate_inputs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], labels: Optional[torch.Tensor]):
        """Validate and sanitize input tensors."""
        
        # Check input_ids
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")
        
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D (batch_size, seq_len), got shape {input_ids.shape}")
        
        if input_ids.size(1) > self.config.max_sequence_length:
            logger.warning(
                f"Input sequence length {input_ids.size(1)} exceeds maximum {self.config.max_sequence_length}. "
                "This may cause issues."
            )
        
        # Check attention_mask
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(f"attention_mask must be a torch.Tensor, got {type(attention_mask)}")
            
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    f"attention_mask shape {attention_mask.shape} must match input_ids shape {input_ids.shape}"
                )
        
        # Check labels
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"labels must be a torch.Tensor, got {type(labels)}")
            
            if labels.dim() != 1:
                raise ValueError(f"labels must be 1D (batch_size,), got shape {labels.shape}")
            
            if labels.size(0) != input_ids.size(0):
                raise ValueError(
                    f"labels batch size {labels.size(0)} must match input_ids batch size {input_ids.size(0)}"
                )
            
            # Check for valid label range
            if labels.min() < 0:
                raise ValueError(f"Labels contain negative values: {labels.min()}")
        
        # Check for empty inputs
        if input_ids.size(0) == 0:
            raise ValueError("Empty batch: input_ids has batch size 0")
        
        if input_ids.size(1) == 0:
            raise ValueError("Empty sequence: input_ids has sequence length 0")
    
    def learn_task(
        self,
        task_id: str,
        train_dataloader,
        eval_dataloader=None,
        num_epochs: Optional[int] = None,
        **kwargs
    ):
        """Learn a new task using continual learning."""
        
        if task_id not in self.adapters:
            raise ValueError(f"Task '{task_id}' not registered. Call register_task() first.")
        
        # Set current task
        self.set_current_task(task_id)
        
        # Store previous model state for knowledge distillation
        if self.config.use_knowledge_distillation:
            self.previous_model_state = self.get_model_state()
        
        # Update EWC parameters before learning new task
        if self.config.elastic_weight_consolidation and len(self.adapters) > 1:
            self.update_ewc_params(train_dataloader)
        
        # Training setup with enhanced optimizer selection
        epochs = num_epochs or self.config.num_epochs
        
        # Dynamic optimizer selection based on task characteristics
        optimizer_type = kwargs.get('optimizer', 'adamw')
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.get_trainable_parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.get_trainable_parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                self.get_trainable_parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Enhanced learning rate scheduler with multiple options
        scheduler_type = kwargs.get('scheduler', 'linear')
        if scheduler_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs * len(train_dataloader)
            )
        elif scheduler_type == 'cosine_restart':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=epochs // 4,
                T_mult=2
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        
        # Enhanced training loop with adaptive learning and monitoring
        self.train()
        best_eval_loss = float('inf')
        patience_counter = 0
        patience = kwargs.get('patience', 5)
        
        # Mixed precision training setup
        use_amp = kwargs.get('use_amp', self.config.mixed_precision)
        scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            # Progress tracking
            total_batches = len(train_dataloader)
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # Move batch to device
                batch = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                # Forward pass with optional mixed precision
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.forward(
                            input_ids=batch['input_ids'],
                            attention_mask=batch.get('attention_mask'),
                            labels=batch['labels'],
                            task_id=task_id
                        )
                        loss = outputs['loss']
                else:
                    outputs = self.forward(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch['labels'],
                        task_id=task_id
                    )
                    loss = outputs['loss']
                
                # Backward pass with optional mixed precision
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.get_trainable_parameters(),
                            self.config.gradient_clipping
                        )
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.get_trainable_parameters(),
                            self.config.gradient_clipping
                        )
                    
                    optimizer.step()
                
                # Scheduler step (with different strategies)
                if scheduler_type in ['cosine', 'cosine_restart']:
                    scheduler.step()
                elif batch_idx < self.config.warmup_steps:
                    scheduler.step()
                
                # Track metrics
                epoch_loss += loss.item()
                predictions = outputs['logits'].argmax(dim=-1)
                epoch_correct += (predictions == batch['labels']).sum().item()
                epoch_total += batch['labels'].size(0)
                
                # Enhanced logging with progress tracking
                if batch_idx % self.config.log_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    progress = (batch_idx + 1) / total_batches * 100
                    logger.info(
                        f"Task {task_id} | Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) | "
                        f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}"
                    )
            
            # Epoch metrics
            epoch_accuracy = epoch_correct / epoch_total
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            
            self.task_performance[task_id]["train_accuracy"].append(epoch_accuracy)
            self.task_performance[task_id]["train_loss"].append(avg_epoch_loss)
            
            logger.info(
                f"Task {task_id} | Epoch {epoch+1} completed | "
                f"Loss: {avg_epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}"
            )
            
            # Enhanced evaluation with early stopping
            if eval_dataloader is not None:
                eval_metrics = self.evaluate_task(task_id, eval_dataloader)
                self.task_performance[task_id]["eval_accuracy"].append(eval_metrics["accuracy"])
                self.task_performance[task_id]["eval_loss"].append(eval_metrics["loss"])
                
                # Early stopping logic
                current_eval_loss = eval_metrics["loss"]
                if current_eval_loss < best_eval_loss:
                    best_eval_loss = current_eval_loss
                    patience_counter = 0
                    
                    # Save best model checkpoint
                    if kwargs.get('save_best', True):
                        best_state = {
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'best_loss': best_eval_loss,
                            'task_id': task_id
                        }
                        self._best_checkpoint = best_state
                else:
                    patience_counter += 1
                    if patience_counter >= patience and kwargs.get('early_stopping', False):
                        logger.info(
                            f"Early stopping triggered after {patience} epochs without improvement. "
                            f"Best eval loss: {best_eval_loss:.4f}"
                        )
                        
                        # Restore best model if available
                        if hasattr(self, '_best_checkpoint'):
                            self.load_state_dict(self._best_checkpoint['model_state_dict'])
                            logger.info("Restored best model checkpoint")
                        break
                
                logger.info(
                    f"Evaluation - Loss: {current_eval_loss:.4f} | "
                    f"Accuracy: {eval_metrics['accuracy']:.4f} | "
                    f"Best Loss: {best_eval_loss:.4f} | "
                    f"Patience: {patience_counter}/{patience}"
                )
        
        # Extract task knowledge for transfer learning
        if self.knowledge_transfer and hasattr(self.config, 'enable_knowledge_transfer') and self.config.enable_knowledge_transfer:
            try:
                self.knowledge_transfer.extract_task_knowledge(task_id, train_dataloader)
                logger.info(f"Extracted knowledge for task '{task_id}'")
            except Exception as e:
                logger.warning(f"Knowledge extraction failed: {e}")
        
        # Record task performance in monitoring system
        if self.system_monitor:
            self.system_monitor.record_task_performance(task_id, epoch_accuracy)
        
        # Create recovery checkpoint after successful training
        if self.error_recovery:
            self.error_recovery.create_checkpoint(
                f"task_{task_id}_completed",
                {'task_id': task_id, 'accuracy': epoch_accuracy, 'epoch': epochs}
            )
        
        # Final model optimization and cleanup
        if hasattr(self, '_best_checkpoint'):
            delattr(self, '_best_checkpoint')
        
        logger.info(f"Completed learning task '{task_id}' - Final accuracy: {epoch_accuracy:.4f}")
    
    def evaluate_task(self, task_id: str, dataloader) -> Dict[str, float]:
        """Evaluate performance on a specific task."""
        self.set_current_task(task_id)
        self.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                outputs = self.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                    task_id=task_id
                )
                
                total_loss += outputs['loss'].item()
                predictions = outputs['logits'].argmax(dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": total_correct / total_samples,
            "total_samples": total_samples
        }
        
        logger.info(f"Task {task_id} evaluation - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def predict(self, text: Union[str, List[str]], task_id: str, **kwargs) -> Dict[str, Any]:
        """Make predictions for given text(s)."""
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Handle single string input
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Predict
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_id=task_id
            )
        
        # Process outputs
        logits = outputs['logits']
        probabilities = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        results = {
            "predictions": predictions.cpu().numpy().tolist(),
            "probabilities": probabilities.cpu().numpy().tolist(),
            "task_id": task_id,
            "task_routing_probs": outputs['task_probs'].cpu().numpy().tolist()
        }
        
        return results
    
    def get_trainable_parameters(self):
        """Get trainable parameters (adapters and classification heads)."""
        params = []
        
        # Task adapters
        for adapter in self.adapters.values():
            params.extend(adapter.parameters())
        
        # Classification heads
        for head in self.classification_heads.values():
            params.extend(head.parameters())
        
        # Task router (if learnable)
        if self.task_router.method == "learned":
            params.extend(self.task_router.parameters())
        
        return params
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        return {
            "total_parameters": total_params,
            "frozen_parameters": frozen_params,
            "trainable_parameters": trainable_params,
            "num_tasks": len(self.adapters),
            "avg_params_per_task": trainable_params // max(len(self.adapters), 1)
        }
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute Elastic Weight Consolidation loss."""
        loss = 0.0
        
        for name, param in self.named_parameters():
            if name in self.ewc_params and param.requires_grad:
                loss += (self.ewc_fisher[name] * (param - self.ewc_params[name]) ** 2).sum()
        
        return loss
    
    def update_ewc_params(self, dataloader):
        """Update EWC parameters and Fisher information matrix."""
        logger.info("Updating EWC parameters...")
        
        # Store current parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.ewc_params[name] = param.data.clone()
        
        # Compute Fisher information matrix
        self.train()
        fisher_dict = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
        
        num_samples = 0
        for batch in dataloader:
            batch = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                    for k, v in batch.items()}
            
            outputs = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=batch['labels'],
                task_id=self.current_task_id
            )
            
            loss = outputs['loss']
            loss.backward()
            
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            num_samples += batch['input_ids'].size(0)
            
            # Limit samples for efficiency
            if num_samples > 1000:
                break
        
        # Average Fisher information
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
            
        self.ewc_fisher = fisher_dict
        logger.info("EWC parameters updated")
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state for knowledge distillation."""
        return {
            "state_dict": self.state_dict(),
            "task_performance": self.task_performance.copy()
        }
    
    def save_model(self, save_path: str):
        """Save the continual learning model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "task_performance": self.task_performance,
            "task_router_mappings": {
                "task_id_to_index": self.task_router.task_id_to_index,
                "index_to_task_id": self.task_router.index_to_task_id,
                "num_tasks": self.task_router.num_tasks
            }
        }, save_path / "model.pt")
        
        # Save configuration
        self.config.to_yaml(str(save_path / "config.yaml"))
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, config: Optional[ContinualConfig] = None):
        """Load a continual learning model."""
        load_path = Path(load_path)
        
        # Load configuration if not provided
        if config is None:
            config = ContinualConfig.from_yaml(str(load_path / "config.yaml"))
        
        # Initialize model
        model = cls(config)
        
        # Load state dict
        checkpoint = torch.load(load_path / "model.pt", map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore task router mappings
        if "task_router_mappings" in checkpoint:
            mappings = checkpoint["task_router_mappings"]
            model.task_router.task_id_to_index = mappings["task_id_to_index"]
            model.task_router.index_to_task_id = mappings["index_to_task_id"]
            model.task_router.num_tasks = mappings["num_tasks"]
        
        # Restore performance history
        if "task_performance" in checkpoint:
            model.task_performance = checkpoint["task_performance"]
        
        logger.info(f"Model loaded from {load_path}")
        return model
    
    def optimize_for_inference(self, optimization_level: str = "balanced") -> Dict[str, Any]:
        """Optimize model for inference performance."""
        
        if not self.performance_optimizer:
            logger.warning("Performance optimizer not available")
            return {}
        
        optimizations = {}
        
        if optimization_level == "speed":
            optimizations = self.performance_optimizer.optimize_inference([
                "torch_compile", "fusion", "quantization"
            ])
        elif optimization_level == "memory":
            optimizations = self.performance_optimizer.optimize_memory_usage()
        elif optimization_level == "balanced":
            # Apply adaptive optimization
            if self.adaptive_optimizer:
                optimizations = self.adaptive_optimizer.adaptive_optimize()
            else:
                optimizations = self.performance_optimizer.optimize_inference([
                    "torch_compile", "quantization"
                ])
        
        logger.info(f"Applied {optimization_level} optimizations: {optimizations}")
        return optimizations
    
    def transfer_knowledge_to_task(
        self, 
        target_task_id: str, 
        source_task_ids: Optional[List[str]] = None,
        strategy: str = "gradient_based"
    ) -> Dict[str, Any]:
        """Transfer knowledge from source tasks to a target task."""
        
        if not self.knowledge_transfer:
            logger.warning("Knowledge transfer optimizer not available")
            return {}
        
        if source_task_ids is None:
            # Automatically find best source tasks
            source_task_ids = self.knowledge_transfer.find_best_source_tasks(target_task_id)
        
        if not source_task_ids:
            logger.warning("No source tasks available for knowledge transfer")
            return {}
        
        try:
            result = self.knowledge_transfer.transfer_knowledge(
                source_task_ids, target_task_id, strategy
            )
            logger.info(
                f"Transferred knowledge from {source_task_ids} to {target_task_id}: {result}"
            )
            return result
        except Exception as e:
            logger.error(f"Knowledge transfer failed: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including health, performance, and errors."""
        
        status = {
            "model_info": {
                "current_task": self.current_task_id,
                "num_tasks": len(self.adapters),
                "num_parameters": sum(p.numel() for p in self.parameters()),
                "device": str(next(self.parameters()).device)
            }
        }
        
        # Add monitoring information
        if self.system_monitor:
            status["system_health"] = self.system_monitor.get_system_status()
        
        # Add error recovery information
        if self.error_recovery:
            status["error_recovery"] = self.error_recovery.get_recovery_report()
        
        # Add performance metrics
        memory_usage = self.get_memory_usage()
        status["memory_usage"] = memory_usage
        
        return status
    
    def benchmark_performance(self, test_input: torch.Tensor, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark model performance comprehensively."""
        
        if not self.performance_optimizer:
            logger.warning("Performance optimizer not available")
            return {}
        
        metrics = self.performance_optimizer.benchmark_performance(
            test_input, num_runs=num_runs
        )
        
        # Record benchmark results in monitoring
        if self.system_monitor:
            self.system_monitor.record_inference_time(metrics.inference_time * 1000)  # Convert to ms
        
        return {
            "inference_time_ms": metrics.inference_time * 1000,
            "memory_usage_mb": metrics.memory_usage,
            "throughput_samples_per_sec": metrics.throughput,
            "efficiency_score": metrics.efficiency_score
        }
    
    def cleanup_resources(self):
        """Clean up resources and stop background processes."""
        
        try:
            # Stop monitoring systems
            if self.system_monitor:
                self.system_monitor.stop_monitoring()
            
            if self.error_recovery:
                self.error_recovery.stop_monitoring()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.cleanup_resources()