"""
Distributed training framework for continual learning at scale.
Supports multi-GPU, multi-node, and federated learning scenarios.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
import os
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
import queue

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = True
    broadcast_buffers: bool = True


class DistributedTrainingManager:
    """Manages distributed training for continual learning."""
    
    def __init__(self, model, config: DistributedConfig):
        self.model = model
        self.config = config
        self.is_initialized = False
        self.ddp_model = None
        
    def initialize_distributed(self):
        """Initialize distributed training environment."""
        
        if self.is_initialized:
            logger.warning("Distributed training already initialized")
            return
        
        # Set environment variables
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ["RANK"] = str(self.config.rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        # Set local rank for CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self.model = self.model.to(f"cuda:{self.config.local_rank}")
        
        # Wrap model with DDP
        self.ddp_model = DDP(
            self.model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers
        )
        
        self.is_initialized = True
        logger.info(
            f"Distributed training initialized - Rank: {self.config.rank}/{self.config.world_size}"
        )
    
    def get_distributed_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """Create distributed dataloader."""
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return dataloader, sampler
    
    def distributed_learn_task(
        self,
        task_id: str,
        train_dataset,
        eval_dataset=None,
        batch_size: int = 32,
        num_epochs: int = 10,
        **kwargs
    ):
        """Learn task in distributed manner."""
        
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")
        
        # Create distributed dataloaders
        train_dataloader, train_sampler = self.get_distributed_dataloader(
            train_dataset, batch_size, shuffle=True
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader, _ = self.get_distributed_dataloader(
                eval_dataset, batch_size, shuffle=False
            )
        
        # Ensure model is registered for task
        if hasattr(self.model, 'register_task'):
            num_labels = kwargs.get('num_labels', 2)
            self.model.register_task(task_id, num_labels)
        
        # Distributed training loop
        for epoch in range(num_epochs):
            # Set epoch for sampler (important for shuffling)
            train_sampler.set_epoch(epoch)
            
            # Train epoch
            self._distributed_train_epoch(
                task_id, train_dataloader, epoch, num_epochs, **kwargs
            )
            
            # Evaluate if requested
            if eval_dataloader and self.config.rank == 0:
                metrics = self._distributed_eval_epoch(task_id, eval_dataloader)
                logger.info(f"Epoch {epoch + 1} evaluation: {metrics}")
        
        # Synchronize all processes
        self.synchronize()
        
        if self.config.rank == 0:
            logger.info(f"Distributed training completed for task '{task_id}'")
    
    def _distributed_train_epoch(
        self,
        task_id: str,
        dataloader,
        epoch: int,
        total_epochs: int,
        **kwargs
    ):
        """Train single epoch in distributed manner."""
        
        self.ddp_model.train()
        
        # Setup optimizer for distributed training
        optimizer = self._get_distributed_optimizer(**kwargs)
        
        total_loss = 0.0
        step_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if torch.cuda.is_available():
                batch = {
                    k: v.to(f"cuda:{self.config.local_rank}") if hasattr(v, 'to') else v
                    for k, v in batch.items()
                }
            
            # Forward pass
            outputs = self.ddp_model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=batch['labels'],
                task_id=task_id
            )
            
            loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                
                # Gradient clipping
                if hasattr(self.model.config, 'gradient_clipping') and self.model.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.ddp_model.parameters(),
                        self.model.config.gradient_clipping
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
            
            total_loss += loss.item()
            
            # Logging on rank 0
            if self.config.rank == 0 and batch_idx % 50 == 0:
                logger.info(
                    f"Rank {self.config.rank} | Task {task_id} | "
                    f"Epoch {epoch + 1}/{total_epochs} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Synchronize and aggregate metrics
        avg_loss = self._all_reduce_metric(total_loss / len(dataloader))
        
        if self.config.rank == 0:
            logger.info(
                f"Distributed training epoch {epoch + 1} completed - "
                f"Average loss: {avg_loss:.4f}"
            )
    
    def _distributed_eval_epoch(self, task_id: str, dataloader) -> Dict[str, float]:
        """Evaluate single epoch in distributed manner."""
        
        self.ddp_model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if torch.cuda.is_available():
                    batch = {
                        k: v.to(f"cuda:{self.config.local_rank}") if hasattr(v, 'to') else v
                        for k, v in batch.items()
                    }
                
                outputs = self.ddp_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                    task_id=task_id
                )
                
                total_loss += outputs['loss'].item()
                predictions = outputs['logits'].argmax(dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        # Aggregate metrics across all processes
        avg_loss = self._all_reduce_metric(total_loss / len(dataloader))
        total_correct = self._all_reduce_metric(total_correct, op='sum')
        total_samples = self._all_reduce_metric(total_samples, op='sum')
        
        accuracy = total_correct / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_samples": total_samples
        }
    
    def _get_distributed_optimizer(self, **kwargs):
        """Get optimizer for distributed training."""
        
        lr = kwargs.get('learning_rate', self.model.config.learning_rate)
        weight_decay = kwargs.get('weight_decay', getattr(self.model.config, 'weight_decay', 0.01))
        
        return torch.optim.AdamW(
            self.ddp_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def _all_reduce_metric(self, value: float, op: str = 'mean') -> float:
        """All-reduce metric across distributed processes."""
        
        tensor = torch.tensor(value, device=f"cuda:{self.config.local_rank}" if torch.cuda.is_available() else "cpu")
        
        if op == 'mean':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.config.world_size
        elif op == 'sum':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        return tensor.item()
    
    def synchronize(self):
        """Synchronize all distributed processes."""
        if self.is_initialized:
            dist.barrier()
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Distributed training cleanup completed")


class FederatedLearningManager:
    """Federated learning implementation for continual learning."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.client_models = {}
        self.global_state = None
        self.round_metrics = []
    
    def add_client(self, client_id: str, client_data):
        """Add federated learning client."""
        
        # Create client model (copy of global model)
        client_model = type(self.model)(self.model.config)
        client_model.load_state_dict(self.model.state_dict())
        
        self.client_models[client_id] = {
            "model": client_model,
            "data": client_data,
            "metrics": []
        }
        
        logger.info(f"Added federated client: {client_id}")
    
    def federated_learning_round(
        self,
        task_id: str,
        selected_clients: Optional[List[str]] = None,
        local_epochs: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute one round of federated learning."""
        
        if selected_clients is None:
            selected_clients = list(self.client_models.keys())
        
        logger.info(
            f"Starting federated learning round with {len(selected_clients)} clients"
        )
        
        # Client training phase
        client_updates = {}
        client_metrics = {}
        
        for client_id in selected_clients:
            logger.info(f"Training client: {client_id}")
            
            # Train client model
            update, metrics = self._train_client(
                client_id, task_id, local_epochs, **kwargs
            )
            
            client_updates[client_id] = update
            client_metrics[client_id] = metrics
        
        # Server aggregation phase
        self._aggregate_client_updates(client_updates)
        
        # Update client models with new global model
        self._broadcast_global_model(selected_clients)
        
        # Aggregate and store round metrics
        round_metrics = self._aggregate_round_metrics(client_metrics)
        self.round_metrics.append(round_metrics)
        
        logger.info(f"Federated learning round completed: {round_metrics}")
        return round_metrics
    
    def _train_client(
        self,
        client_id: str,
        task_id: str,
        local_epochs: int,
        **kwargs
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Train individual client model."""
        
        client_info = self.client_models[client_id]
        client_model = client_info["model"]
        client_data = client_info["data"]
        
        # Store initial state
        initial_state = {
            name: param.clone() for name, param in client_model.named_parameters()
        }
        
        # Train client model
        client_model.train()
        
        # Create dataloader for client data
        dataloader = torch.utils.data.DataLoader(
            client_data,
            batch_size=kwargs.get('batch_size', 16),
            shuffle=True
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            client_model.parameters(),
            lr=kwargs.get('learning_rate', 1e-4)
        )
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(local_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                
                outputs = client_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                    task_id=task_id
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                predictions = outputs['logits'].argmax(dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        # Calculate parameter updates
        updates = {}
        for name, param in client_model.named_parameters():
            updates[name] = param.data - initial_state[name]
        
        # Calculate client metrics
        metrics = {
            "loss": total_loss / (len(dataloader) * local_epochs),
            "accuracy": total_correct / total_samples,
            "samples": total_samples
        }
        
        client_info["metrics"].append(metrics)
        
        return updates, metrics
    
    def _aggregate_client_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]):
        """Aggregate client updates using federated averaging."""
        
        # Calculate weighted average based on number of samples
        aggregated_updates = {}
        total_samples = 0
        
        # Calculate total samples
        for client_id, updates in client_updates.items():
            client_info = self.client_models[client_id]
            client_samples = client_info["metrics"][-1]["samples"]
            total_samples += client_samples
        
        # Weighted aggregation
        for param_name in next(iter(client_updates.values())).keys():
            weighted_sum = torch.zeros_like(
                next(iter(client_updates.values()))[param_name]
            )
            
            for client_id, updates in client_updates.items():
                client_info = self.client_models[client_id]
                client_samples = client_info["metrics"][-1]["samples"]
                weight = client_samples / total_samples
                
                weighted_sum += weight * updates[param_name]
            
            aggregated_updates[param_name] = weighted_sum
        
        # Apply aggregated updates to global model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_updates:
                    param.data += aggregated_updates[name]
        
        logger.info("Client updates aggregated to global model")
    
    def _broadcast_global_model(self, client_ids: List[str]):
        """Broadcast updated global model to clients."""
        
        global_state = self.model.state_dict()
        
        for client_id in client_ids:
            self.client_models[client_id]["model"].load_state_dict(global_state)
        
        logger.info(f"Global model broadcasted to {len(client_ids)} clients")
    
    def _aggregate_round_metrics(self, client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from all clients in the round."""
        
        total_samples = sum(metrics["samples"] for metrics in client_metrics.values())
        
        # Weighted average of metrics
        avg_loss = sum(
            metrics["loss"] * metrics["samples"] for metrics in client_metrics.values()
        ) / total_samples
        
        avg_accuracy = sum(
            metrics["accuracy"] * metrics["samples"] for metrics in client_metrics.values()
        ) / total_samples
        
        return {
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "total_samples": total_samples,
            "num_clients": len(client_metrics)
        }
    
    def get_federated_metrics(self) -> Dict[str, Any]:
        """Get comprehensive federated learning metrics."""
        
        return {
            "total_rounds": len(self.round_metrics),
            "total_clients": len(self.client_models),
            "round_history": self.round_metrics,
            "client_metrics": {
                client_id: info["metrics"]
                for client_id, info in self.client_models.items()
            }
        }


def launch_distributed_training(
    rank: int,
    world_size: int,
    model_func: Callable,
    train_func: Callable,
    config: DistributedConfig,
    **kwargs
):
    """Launch distributed training process."""
    
    # Update config for current process
    config.rank = rank
    config.world_size = world_size
    config.local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Initialize model
    model = model_func()
    
    # Setup distributed manager
    dist_manager = DistributedTrainingManager(model, config)
    dist_manager.initialize_distributed()
    
    try:
        # Run training function
        train_func(dist_manager, **kwargs)
    finally:
        # Cleanup
        dist_manager.cleanup()


def start_distributed_training(
    world_size: int,
    model_func: Callable,
    train_func: Callable,
    config: DistributedConfig,
    **kwargs
):
    """Start multi-process distributed training."""
    
    if world_size > 1:
        mp.spawn(
            launch_distributed_training,
            args=(world_size, model_func, train_func, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single process training
        launch_distributed_training(0, 1, model_func, train_func, config, **kwargs)