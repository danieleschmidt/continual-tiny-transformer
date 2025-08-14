"""
Distributed Continual Learning System for Large-Scale Deployment.
Supports multi-GPU, multi-node continual learning with advanced coordination.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed continual learning."""
    backend: str = "nccl"  # or "gloo"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # Continual learning specific
    task_distribution_strategy: str = "round_robin"  # "round_robin", "expert", "random"
    gradient_accumulation_steps: int = 1
    sync_frequency: int = 100  # Steps between synchronization
    
    # Performance optimization
    use_gradient_compression: bool = True
    use_model_parallelism: bool = False
    use_pipeline_parallelism: bool = False
    mixed_precision: bool = True
    
    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 1000
    enable_elastic_training: bool = False


@dataclass
class TaskAssignment:
    """Assignment of tasks to workers."""
    worker_id: int
    task_ids: List[str]
    load_balance_score: float
    specialization_score: float


class DistributedTaskCoordinator:
    """Coordinates task distribution across multiple workers."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.task_assignments = {}
        self.worker_capabilities = {}
        self.task_completion_history = defaultdict(list)
        self.load_balancing_metrics = defaultdict(dict)
        
    def register_worker(self, worker_id: int, capabilities: Dict[str, Any]):
        """Register a worker with its capabilities."""
        self.worker_capabilities[worker_id] = capabilities
        logger.info(f"Registered worker {worker_id} with capabilities: {capabilities}")
    
    def assign_tasks(self, task_list: List[str]) -> Dict[int, List[str]]:
        """Assign tasks to workers based on strategy."""
        
        if self.config.task_distribution_strategy == "round_robin":
            return self._assign_round_robin(task_list)
        elif self.config.task_distribution_strategy == "expert":
            return self._assign_expert_based(task_list)
        elif self.config.task_distribution_strategy == "random":
            return self._assign_random(task_list)
        else:
            raise ValueError(f"Unknown distribution strategy: {self.config.task_distribution_strategy}")
    
    def _assign_round_robin(self, task_list: List[str]) -> Dict[int, List[str]]:
        """Simple round-robin task assignment."""
        assignments = defaultdict(list)
        
        for i, task_id in enumerate(task_list):
            worker_id = i % self.config.world_size
            assignments[worker_id].append(task_id)
        
        return dict(assignments)
    
    def _assign_expert_based(self, task_list: List[str]) -> Dict[int, List[str]]:
        """Assign tasks based on worker expertise."""
        assignments = defaultdict(list)
        
        # Simple expertise scoring based on past performance
        for task_id in task_list:
            best_worker = 0
            best_score = 0
            
            for worker_id in range(self.config.world_size):
                # Score based on past performance on similar tasks
                history = self.task_completion_history.get(f"{worker_id}_{task_id}", [])
                score = np.mean(history) if history else 0.5  # Default score
                
                # Consider current load
                current_load = len(assignments[worker_id])
                adjusted_score = score * (1.0 / (1.0 + current_load * 0.1))
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_worker = worker_id
            
            assignments[best_worker].append(task_id)
        
        return dict(assignments)
    
    def _assign_random(self, task_list: List[str]) -> Dict[int, List[str]]:
        """Random task assignment with load balancing."""
        assignments = defaultdict(list)
        
        for task_id in task_list:
            # Choose worker with least current load
            worker_loads = {i: len(assignments[i]) for i in range(self.config.world_size)}
            min_load_worker = min(worker_loads, key=worker_loads.get)
            assignments[min_load_worker].append(task_id)
        
        return dict(assignments)
    
    def record_task_completion(
        self,
        worker_id: int,
        task_id: str,
        performance_score: float,
        completion_time: float
    ):
        """Record task completion for future assignment optimization."""
        self.task_completion_history[f"{worker_id}_{task_id}"].append(performance_score)
        self.load_balancing_metrics[worker_id][task_id] = {
            "completion_time": completion_time,
            "performance_score": performance_score,
            "timestamp": time.time()
        }


class DistributedContinualLearner:
    """Main distributed continual learning coordinator."""
    
    def __init__(self, model, config: DistributedConfig):
        self.model = model
        self.config = config
        self.task_coordinator = DistributedTaskCoordinator(config)
        
        # Distributed training state
        self.is_initialized = False
        self.ddp_model = None
        
        # Synchronization
        self.sync_queue = queue.Queue()
        self.gradient_buffer = {}
        
        # Performance tracking
        self.throughput_metrics = defaultdict(list)
        self.communication_overhead = []
        
    def initialize_distributed(self):
        """Initialize distributed training environment."""
        
        if self.is_initialized:
            logger.warning("Distributed training already initialized")
            return
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            device = torch.device(f"cuda:{self.config.local_rank}")
        else:
            device = torch.device("cpu")
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Wrap model with DDP
        self.ddp_model = DDP(
            self.model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True  # Important for continual learning
        )
        
        # Register worker capabilities
        capabilities = {
            "device_type": device.type,
            "memory_gb": torch.cuda.get_device_properties(device).total_memory / 1e9 if device.type == "cuda" else 8,
            "compute_capability": torch.cuda.get_device_properties(device).major if device.type == "cuda" else 0
        }
        
        self.task_coordinator.register_worker(self.config.rank, capabilities)
        self.is_initialized = True
        
        logger.info(f"Initialized distributed training on rank {self.config.rank}")
    
    def learn_tasks_distributed(
        self,
        task_dataloaders: Dict[str, Any],
        task_configs: Dict[str, Dict[str, Any]],
        global_epochs: int = 10
    ):
        """Learn multiple tasks in a distributed fashion."""
        
        if not self.is_initialized:
            self.initialize_distributed()
        
        # Assign tasks to workers
        task_list = list(task_dataloaders.keys())
        task_assignments = self.task_coordinator.assign_tasks(task_list)
        my_tasks = task_assignments.get(self.config.rank, [])
        
        logger.info(f"Worker {self.config.rank} assigned tasks: {my_tasks}")
        
        # Learn assigned tasks
        for epoch in range(global_epochs):
            epoch_start_time = time.time()
            
            for task_id in my_tasks:
                if task_id not in task_dataloaders:
                    logger.warning(f"Task {task_id} not found in dataloaders")
                    continue
                
                task_start_time = time.time()
                
                # Learn single task
                task_metrics = self._learn_single_task_distributed(
                    task_id,
                    task_dataloaders[task_id],
                    task_configs.get(task_id, {})
                )
                
                task_completion_time = time.time() - task_start_time
                
                # Record completion
                self.task_coordinator.record_task_completion(
                    self.config.rank,
                    task_id,
                    task_metrics.get("accuracy", 0.0),
                    task_completion_time
                )
                
                # Synchronize periodically
                if epoch % self.config.sync_frequency == 0:
                    self._synchronize_models()
            
            epoch_time = time.time() - epoch_start_time
            self.throughput_metrics["epoch_time"].append(epoch_time)
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Final synchronization
        self._synchronize_models()
        
        logger.info("Distributed continual learning completed")
    
    def _learn_single_task_distributed(
        self,
        task_id: str,
        dataloader,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn a single task in distributed setting."""
        
        # Set up distributed sampler if needed
        if hasattr(dataloader.dataset, '__len__'):
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank
            )
        else:
            sampler = None
        
        # Create distributed dataloader
        from torch.utils.data import DataLoader
        
        if sampler:
            dist_dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=sampler,
                num_workers=getattr(dataloader, 'num_workers', 0),
                collate_fn=getattr(dataloader, 'collate_fn', None)
            )
        else:
            dist_dataloader = dataloader
        
        # Training setup
        optimizer = torch.optim.AdamW(
            self.ddp_model.parameters(),
            lr=task_config.get("learning_rate", 2e-5)
        )
        
        # Training loop
        self.ddp_model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dist_dataloader):
            # Move batch to device
            batch = {k: v.to(self.ddp_model.device) if hasattr(v, 'to') else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.ddp_model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=batch['labels'],
                task_id=task_id
            )
            
            loss = outputs['loss']
            
            # Backward pass with gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient compression if enabled
                if self.config.use_gradient_compression:
                    self._compress_gradients()
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            predictions = outputs['logits'].argmax(dim=-1)
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)
            
            # Periodic synchronization
            if batch_idx % self.config.sync_frequency == 0:
                self._exchange_gradients()
        
        # Calculate final metrics
        avg_loss = total_loss / len(dist_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # All-reduce metrics across workers
        metrics_tensor = torch.tensor([avg_loss, accuracy], device=self.ddp_model.device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= self.config.world_size
        
        final_metrics = {
            "loss": metrics_tensor[0].item(),
            "accuracy": metrics_tensor[1].item(),
            "samples_processed": total_samples
        }
        
        logger.info(f"Task {task_id} - Loss: {final_metrics['loss']:.4f}, "
                   f"Accuracy: {final_metrics['accuracy']:.4f}")
        
        return final_metrics
    
    def _synchronize_models(self):
        """Synchronize model parameters across workers."""
        
        start_time = time.time()
        
        # Synchronize all parameters
        for param in self.ddp_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.config.world_size
        
        # Synchronize model-specific state (e.g., task router mappings)
        if hasattr(self.model, 'task_router'):
            # Broadcast task router state from rank 0
            if self.config.rank == 0:
                router_state = {
                    "task_id_to_index": self.model.task_router.task_id_to_index,
                    "index_to_task_id": self.model.task_router.index_to_task_id,
                    "num_tasks": self.model.task_router.num_tasks
                }
                
                # Serialize and broadcast
                serialized_state = pickle.dumps(router_state)
                state_tensor = torch.tensor(
                    list(serialized_state),
                    dtype=torch.uint8,
                    device=self.ddp_model.device
                )
                
                # Broadcast size first
                size_tensor = torch.tensor(len(serialized_state), device=self.ddp_model.device)
                dist.broadcast(size_tensor, src=0)
                
                # Broadcast state
                dist.broadcast(state_tensor, src=0)
            else:
                # Receive state
                size_tensor = torch.tensor(0, device=self.ddp_model.device)
                dist.broadcast(size_tensor, src=0)
                
                state_tensor = torch.zeros(
                    size_tensor.item(),
                    dtype=torch.uint8,
                    device=self.ddp_model.device
                )
                dist.broadcast(state_tensor, src=0)
                
                # Deserialize and apply
                serialized_state = bytes(state_tensor.cpu().numpy())
                router_state = pickle.loads(serialized_state)
                
                self.model.task_router.task_id_to_index = router_state["task_id_to_index"]
                self.model.task_router.index_to_task_id = router_state["index_to_task_id"]
                self.model.task_router.num_tasks = router_state["num_tasks"]
        
        sync_time = time.time() - start_time
        self.communication_overhead.append(sync_time)
        
        logger.debug(f"Model synchronization completed in {sync_time:.3f}s")
    
    def _compress_gradients(self):
        """Apply gradient compression to reduce communication overhead."""
        
        # Simple top-k compression
        compression_ratio = 0.1  # Keep top 10% of gradients
        
        for param in self.ddp_model.parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Flatten gradient
                flat_grad = grad.view(-1)
                
                # Find top-k elements
                k = max(1, int(len(flat_grad) * compression_ratio))
                topk_values, topk_indices = torch.topk(torch.abs(flat_grad), k)
                
                # Create sparse gradient
                compressed_grad = torch.zeros_like(flat_grad)
                compressed_grad[topk_indices] = flat_grad[topk_indices]
                
                # Reshape back
                param.grad.data = compressed_grad.view(grad.shape)
    
    def _exchange_gradients(self):
        """Exchange gradients with other workers for advanced synchronization."""
        
        # Simple all-reduce for now
        # Could implement more sophisticated algorithms like:
        # - Ring all-reduce
        # - Tree all-reduce
        # - Hierarchical all-reduce
        
        for param in self.ddp_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.config.world_size
    
    def save_distributed_checkpoint(self, checkpoint_path: str):
        """Save distributed checkpoint."""
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if self.config.rank == 0:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "distributed_config": asdict(self.config),
                "task_assignments": self.task_coordinator.task_assignments,
                "performance_metrics": dict(self.throughput_metrics)
            }, checkpoint_path / "distributed_checkpoint.pt")
        
        # Save rank-specific state
        rank_checkpoint = {
            "rank": self.config.rank,
            "local_state": getattr(self.model, 'local_state', {}),
            "completion_history": dict(self.task_coordinator.task_completion_history)
        }
        
        torch.save(rank_checkpoint, checkpoint_path / f"rank_{self.config.rank}_checkpoint.pt")
        
        # Synchronize to ensure all saves complete
        dist.barrier()
        
        if self.config.rank == 0:
            logger.info(f"Distributed checkpoint saved to {checkpoint_path}")
    
    def load_distributed_checkpoint(self, checkpoint_path: str):
        """Load distributed checkpoint."""
        
        checkpoint_path = Path(checkpoint_path)
        
        # Load main checkpoint
        main_checkpoint = torch.load(
            checkpoint_path / "distributed_checkpoint.pt",
            map_location=self.ddp_model.device
        )
        
        self.model.load_state_dict(main_checkpoint["model_state_dict"])
        
        # Load rank-specific state
        rank_checkpoint_file = checkpoint_path / f"rank_{self.config.rank}_checkpoint.pt"
        if rank_checkpoint_file.exists():
            rank_checkpoint = torch.load(rank_checkpoint_file, map_location=self.ddp_model.device)
            # Restore local state if available
            if "local_state" in rank_checkpoint:
                if hasattr(self.model, 'local_state'):
                    self.model.local_state = rank_checkpoint["local_state"]
        
        dist.barrier()
        
        logger.info(f"Distributed checkpoint loaded from {checkpoint_path}")
    
    def get_distributed_metrics(self) -> Dict[str, Any]:
        """Get comprehensive distributed training metrics."""
        
        # Gather metrics from all workers
        local_metrics = {
            "rank": self.config.rank,
            "throughput": self.throughput_metrics,
            "communication_overhead": np.mean(self.communication_overhead) if self.communication_overhead else 0,
            "task_completions": len(self.task_coordinator.task_completion_history)
        }
        
        # Serialize local metrics
        serialized_metrics = json.dumps(local_metrics).encode()
        
        # Gather all metrics to rank 0
        if self.config.rank == 0:
            all_metrics = [None] * self.config.world_size
            all_metrics[0] = local_metrics
            
            for rank in range(1, self.config.world_size):
                # This would need proper implementation for gathering variable-length data
                pass
            
            return {
                "world_size": self.config.world_size,
                "total_communication_overhead": np.mean([
                    m.get("communication_overhead", 0) for m in all_metrics if m
                ]),
                "worker_metrics": all_metrics
            }
        else:
            return local_metrics
    
    def cleanup_distributed(self):
        """Clean up distributed training resources."""
        
        if dist.is_initialized():
            dist.destroy_process_group()
        
        self.is_initialized = False
        
        logger.info("Distributed training cleanup completed")


# Utility functions for distributed training
def setup_distributed_training(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl"
) -> DistributedConfig:
    """Set up distributed training configuration."""
    
    import os
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    if torch.cuda.is_available():
        os.environ['LOCAL_RANK'] = str(rank % torch.cuda.device_count())
        local_rank = rank % torch.cuda.device_count()
    else:
        local_rank = 0
    
    config = DistributedConfig(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port
    )
    
    return config


def launch_distributed_training(
    training_function: Callable,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355"
):
    """Launch distributed training across multiple processes."""
    
    mp.spawn(
        training_function,
        args=(world_size, master_addr, master_port),
        nprocs=world_size,
        join=True
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def distributed_training_example(rank, world_size, master_addr, master_port):
        """Example distributed training function."""
        
        # Set up configuration
        config = setup_distributed_training(rank, world_size, master_addr, master_port)
        
        # Mock model for testing
        from torch import nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)
            
            def forward(self, x, **kwargs):
                return {"logits": self.linear(x), "loss": torch.tensor(0.5)}
        
        model = SimpleModel()
        
        # Initialize distributed learner
        learner = DistributedContinualLearner(model, config)
        learner.initialize_distributed()
        
        print(f"Rank {rank} initialized successfully")
        
        # Cleanup
        learner.cleanup_distributed()
    
    # Test with 2 processes
    if torch.cuda.device_count() >= 2:
        launch_distributed_training(distributed_training_example, 2)
    else:
        print("Distributed training requires multiple GPUs")