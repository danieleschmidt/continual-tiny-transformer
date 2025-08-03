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
from ..adapters.activation import ActivationAdapter
from ..tasks.manager import TaskManager
from ..utils.knowledge_distillation import KnowledgeDistillation


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
        
        # Create task-specific adapter
        if self.config.adaptation_method == "activation":
            adapter = ActivationAdapter(
                hidden_size=self.base_model.config.hidden_size,
                adapter_size=64,  # Configurable
                num_layers=self.base_model.config.num_hidden_layers
            )
        else:
            raise NotImplementedError(f"Adaptation method '{self.config.adaptation_method}' not implemented")
        
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
        """Forward pass through the continual transformer."""
        
        # Use current task if not specified
        if task_id is None:
            task_id = self.current_task_id
        
        if task_id is None:
            raise ValueError("No task_id specified and no current task set")
        
        # Base transformer forward pass
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = base_outputs.last_hidden_state
        
        # Task routing (for analysis and potential multi-task scenarios)
        task_probs, predicted_task_indices = self.task_router(hidden_states, task_id)
        
        # Apply task-specific adapter
        if task_id in self.adapters:
            adapted_states = self.adapters[task_id](hidden_states)
        else:
            adapted_states = hidden_states
        
        # Global pooling for classification
        pooled_output = adapted_states.mean(dim=1)
        
        # Task-specific classification
        logits = self.classification_heads[task_id](pooled_output)
        
        outputs = {
            "logits": logits,
            "hidden_states": adapted_states,
            "task_probs": task_probs,
            "predicted_task_indices": predicted_task_indices,
            "pooled_output": pooled_output
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            # Add knowledge distillation loss if applicable
            if (self.config.use_knowledge_distillation and 
                hasattr(self, 'knowledge_distillation') and
                self.previous_model_state is not None):
                
                kd_loss = self.knowledge_distillation.compute_loss(
                    student_logits=logits,
                    teacher_logits=self.previous_model_state.get('logits'),
                    temperature=self.config.temperature
                )
                loss = loss + self.config.knowledge_distillation_alpha * kd_loss
                outputs["kd_loss"] = kd_loss
            
            # Add EWC loss if applicable
            if self.config.elastic_weight_consolidation and self.ewc_params:
                ewc_loss = self.compute_ewc_loss()
                loss = loss + self.config.ewc_lambda * ewc_loss
                outputs["ewc_loss"] = ewc_loss
            
            outputs["loss"] = loss
        
        return outputs if return_dict else (logits,)
    
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
        
        # Training setup
        epochs = num_epochs or self.config.num_epochs
        optimizer = torch.optim.AdamW(
            self.get_trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps
        )
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # Move batch to device
                batch = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                    task_id=task_id
                )
                
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.get_trainable_parameters(),
                        self.config.gradient_clipping
                    )
                
                optimizer.step()
                if batch_idx < self.config.warmup_steps:
                    scheduler.step()
                
                # Track metrics
                epoch_loss += loss.item()
                predictions = outputs['logits'].argmax(dim=-1)
                epoch_correct += (predictions == batch['labels']).sum().item()
                epoch_total += batch['labels'].size(0)
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    logger.info(
                        f"Task {task_id} | Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx} | Loss: {loss.item():.4f}"
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
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate_task(task_id, eval_dataloader)
                self.task_performance[task_id]["eval_accuracy"].append(eval_metrics["accuracy"])
                self.task_performance[task_id]["eval_loss"].append(eval_metrics["loss"])
        
        logger.info(f"Completed learning task '{task_id}'")
    
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