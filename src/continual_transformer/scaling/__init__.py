"""Scaling and distributed computing modules for continual learning."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class DistributedTrainingManager:
    """Manager for distributed training across multiple GPUs/nodes."""
    
    def __init__(
        self, 
        model,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        backend: str = "nccl"
    ):
        self.model = model
        self.backend = backend
        self.is_distributed = False
        self.rank = rank
        self.world_size = world_size
        
        # Auto-detect distributed environment
        if rank is None:
            self.rank = int(os.environ.get("RANK", 0))
        if world_size is None:
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if self.world_size > 1:
            self._setup_distributed()
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        try:
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method="env://",
                    rank=self.rank,
                    world_size=self.world_size
                )
            
            # Set CUDA device for current process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                device = torch.device(f"cuda:{self.local_rank}")
                self.model = self.model.to(device)
                
                # Wrap model with DDP
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True
                )
            else:
                # CPU distributed training
                self.model = DDP(self.model, find_unused_parameters=True)
            
            self.is_distributed = True
            logger.info(f"Distributed training setup complete - Rank: {self.rank}/{self.world_size}")
            
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
            self.is_distributed = False
    
    def create_distributed_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """Create distributed dataloader."""
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )
    
    def all_reduce_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Average metrics across all processes."""
        if not self.is_distributed:
            return metrics
        
        reduced_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                value = torch.tensor(float(value), device=self._get_device())
            
            if isinstance(value, torch.Tensor):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                reduced_metrics[key] = value / self.world_size
            else:
                reduced_metrics[key] = value
        
        return reduced_metrics
    
    def barrier(self):
        """Synchronization barrier."""
        if self.is_distributed:
            dist.barrier()
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed training cleanup complete")
    
    def _get_device(self):
        """Get current device."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

class AsyncInferenceEngine:
    """Asynchronous inference engine for high-throughput serving."""
    
    def __init__(
        self,
        model,
        max_batch_size: int = 32,
        max_workers: int = 4,
        queue_timeout: float = 0.1
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.queue_timeout = queue_timeout
        
        self.request_queue = asyncio.Queue()
        self.response_futures = {}
        self.is_running = False
        self.worker_tasks = []
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    
    async def start(self):
        """Start the async inference engine."""
        if self.is_running:
            logger.warning("Async inference engine already running")
            return
        
        self.is_running = True
        
        # Start batch processing workers
        for i in range(self.max_workers):
            task = asyncio.create_task(self._batch_worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Started async inference engine with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the async inference engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Stopped async inference engine")
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Submit prediction request asynchronously."""
        if not self.is_running:
            raise RuntimeError("Async inference engine not running")
        
        # Create future for response
        request_id = id(input_data)
        future = asyncio.Future()
        self.response_futures[request_id] = future
        
        # Add request to queue
        request = {
            "id": request_id,
            "data": input_data,
            "timestamp": time.time()
        }
        
        await self.request_queue.put(request)
        
        # Wait for response with timeout
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Cleanup future
            if request_id in self.response_futures:
                del self.response_futures[request_id]
            raise TimeoutError(f"Prediction timeout after {timeout}s")
        finally:
            # Ensure cleanup
            self.response_futures.pop(request_id, None)
    
    async def _batch_worker(self, worker_id: str):
        """Worker for processing batched requests."""
        logger.info(f"Started batch worker: {worker_id}")
        
        while self.is_running:
            try:
                # Collect batch of requests
                batch_requests = await self._collect_batch()
                
                if not batch_requests:
                    await asyncio.sleep(0.001)  # Small delay when no requests
                    continue
                
                # Process batch
                await self._process_batch(batch_requests, worker_id)
                
            except Exception as e:
                logger.error(f"Batch worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)  # Error recovery delay
    
    async def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collect a batch of requests from the queue."""
        batch = []
        start_time = time.time()
        
        # Try to collect up to max_batch_size requests
        while len(batch) < self.max_batch_size:
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=self.queue_timeout
                )
                batch.append(request)
                
                # Don't wait too long for batch completion
                if time.time() - start_time > 0.01 and batch:  # 10ms max wait
                    break
                    
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch_requests: List[Dict[str, Any]], worker_id: str):
        """Process a batch of requests."""
        if not batch_requests:
            return
        
        try:
            # Prepare batch input
            batch_data = self._prepare_batch_input(batch_requests)
            
            # Run inference in thread pool (to avoid blocking async loop)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.thread_pool,
                self._run_batch_inference,
                batch_data
            )
            
            # Distribute results to futures
            self._distribute_results(batch_requests, results)
            
            logger.debug(f"Worker {worker_id} processed batch of {len(batch_requests)} requests")
            
        except Exception as e:
            logger.error(f"Batch processing error in {worker_id}: {e}")
            # Send error to all requests in batch
            self._distribute_errors(batch_requests, e)
    
    def _prepare_batch_input(self, batch_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare batch input for model inference."""
        # Extract data from requests
        batch_data = []
        for request in batch_requests:
            batch_data.append(request["data"])
        
        return {"batch": batch_data}
    
    def _run_batch_inference(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run batch inference (executed in thread pool)."""
        with torch.no_grad():
            self.model.eval()
            
            # Process each item in batch
            results = []
            for item_data in batch_data["batch"]:
                try:
                    # Single prediction (would be optimized for actual batching)
                    result = self._single_prediction(item_data)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
            
            return results
    
    def _single_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run single prediction (placeholder implementation)."""
        # This would be replaced with actual model inference logic
        text = data.get("text", "")
        task_id = data.get("task_id", "default")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(text, task_id)
        else:
            # Fallback response
            return {
                "predictions": [0],
                "probabilities": [0.5],
                "task_id": task_id
            }
    
    def _distribute_results(
        self, 
        batch_requests: List[Dict[str, Any]], 
        results: List[Dict[str, Any]]
    ):
        """Distribute results to corresponding futures."""
        for request, result in zip(batch_requests, results):
            request_id = request["id"]
            if request_id in self.response_futures:
                future = self.response_futures[request_id]
                if not future.done():
                    future.set_result(result)
    
    def _distribute_errors(self, batch_requests: List[Dict[str, Any]], error: Exception):
        """Distribute error to all requests in batch."""
        for request in batch_requests:
            request_id = request["id"]
            if request_id in self.response_futures:
                future = self.response_futures[request_id]
                if not future.done():
                    future.set_exception(error)

class ModelSharding:
    """Model sharding for very large models."""
    
    def __init__(self, model, num_shards: int = 2):
        self.original_model = model
        self.num_shards = num_shards
        self.sharded_models = {}
        self.shard_assignments = {}
        
    def shard_model(self) -> Dict[int, nn.Module]:
        """Shard model across multiple devices/processes."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            logger.warning("Model sharding requires multiple GPUs")
            return {0: self.original_model}
        
        # Simple layer-wise sharding strategy
        layers = list(self.original_model.named_children())
        layers_per_shard = len(layers) // self.num_shards
        
        for shard_id in range(self.num_shards):
            device_id = shard_id % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_id}")
            
            # Create shard with subset of layers
            start_idx = shard_id * layers_per_shard
            end_idx = start_idx + layers_per_shard if shard_id < self.num_shards - 1 else len(layers)
            
            shard_layers = dict(layers[start_idx:end_idx])
            shard_model = nn.ModuleDict(shard_layers).to(device)
            
            self.sharded_models[shard_id] = shard_model
            self.shard_assignments[shard_id] = {
                "device": device,
                "layers": list(shard_layers.keys()),
                "start_idx": start_idx,
                "end_idx": end_idx
            }
        
        logger.info(f"Model sharded into {self.num_shards} shards across {torch.cuda.device_count()} GPUs")
        return self.sharded_models
    
    def forward_sharded(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sharded model."""
        current_x = x
        
        for shard_id in sorted(self.sharded_models.keys()):
            shard_model = self.sharded_models[shard_id]
            device = self.shard_assignments[shard_id]["device"]
            
            # Move data to shard device
            current_x = current_x.to(device)
            
            # Forward through shard
            with torch.cuda.device(device):
                for layer_name in self.shard_assignments[shard_id]["layers"]:
                    layer = getattr(shard_model, layer_name)
                    current_x = layer(current_x)
        
        return current_x

class LoadBalancer:
    """Load balancer for distributed inference."""
    
    def __init__(self, model_instances: List[Any]):
        self.model_instances = model_instances
        self.current_instance = 0
        self.instance_stats = {i: {"requests": 0, "avg_time": 0.0, "errors": 0} 
                              for i in range(len(model_instances))}
        self._lock = threading.Lock()
    
    def get_next_instance(self) -> Tuple[int, Any]:
        """Get next model instance using round-robin with health awareness."""
        with self._lock:
            # Simple round-robin for now (could be enhanced with load-aware selection)
            best_instance = self._select_best_instance()
            instance = self.model_instances[best_instance]
            
            self.current_instance = (best_instance + 1) % len(self.model_instances)
            return best_instance, instance
    
    def _select_best_instance(self) -> int:
        """Select best instance based on performance metrics."""
        # Find instance with lowest error rate and reasonable response time
        best_score = float('inf')
        best_instance = 0
        
        for i, stats in self.instance_stats.items():
            # Simple scoring: error_rate + normalized_avg_time
            total_requests = stats["requests"] or 1
            error_rate = stats["errors"] / total_requests
            normalized_time = stats["avg_time"] / 1000.0  # Normalize to seconds
            
            score = error_rate * 10 + normalized_time  # Weight errors more heavily
            
            if score < best_score:
                best_score = score
                best_instance = i
        
        return best_instance
    
    def record_request(self, instance_id: int, response_time: float, success: bool = True):
        """Record request statistics for load balancing."""
        with self._lock:
            stats = self.instance_stats[instance_id]
            stats["requests"] += 1
            
            # Update average response time
            if stats["requests"] == 1:
                stats["avg_time"] = response_time
            else:
                # Exponential moving average
                stats["avg_time"] = 0.9 * stats["avg_time"] + 0.1 * response_time
            
            if not success:
                stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_requests = sum(stats["requests"] for stats in self.instance_stats.values())
            total_errors = sum(stats["errors"] for stats in self.instance_stats.values())
            
            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": total_errors / max(total_requests, 1),
                "instance_stats": dict(self.instance_stats),
                "num_instances": len(self.model_instances)
            }

class ScalingManager:
    """Main manager for all scaling operations."""
    
    def __init__(self, model, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        
        # Initialize scaling components
        self.distributed_manager = None
        self.async_engine = None
        self.load_balancer = None
        self.model_sharding = None
        
        self.scaling_active = False
    
    def setup_distributed_training(self, **kwargs):
        """Setup distributed training environment."""
        self.distributed_manager = DistributedTrainingManager(self.model, **kwargs)
        logger.info("Distributed training manager initialized")
    
    def setup_async_inference(self, **kwargs):
        """Setup asynchronous inference engine."""
        async_config = {
            "max_batch_size": self.config.get("max_batch_size", 32),
            "max_workers": self.config.get("max_workers", 4),
            **kwargs
        }
        
        self.async_engine = AsyncInferenceEngine(self.model, **async_config)
        logger.info("Async inference engine initialized")
    
    def setup_load_balancing(self, model_instances: List[Any]):
        """Setup load balancing across model instances."""
        self.load_balancer = LoadBalancer(model_instances)
        logger.info(f"Load balancer initialized with {len(model_instances)} instances")
    
    def setup_model_sharding(self, num_shards: int = 2):
        """Setup model sharding for large models."""
        self.model_sharding = ModelSharding(self.model, num_shards)
        sharded_models = self.model_sharding.shard_model()
        logger.info(f"Model sharding setup with {len(sharded_models)} shards")
    
    async def start_scaling_services(self):
        """Start all scaling services."""
        if self.async_engine:
            await self.async_engine.start()
        
        self.scaling_active = True
        logger.info("Scaling services started")
    
    async def stop_scaling_services(self):
        """Stop all scaling services."""
        if self.async_engine:
            await self.async_engine.stop()
        
        if self.distributed_manager:
            self.distributed_manager.cleanup()
        
        self.scaling_active = False
        logger.info("Scaling services stopped")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        status = {
            "scaling_active": self.scaling_active,
            "distributed_training": self.distributed_manager is not None,
            "async_inference": self.async_engine is not None,
            "load_balancing": self.load_balancer is not None,
            "model_sharding": self.model_sharding is not None
        }
        
        # Add detailed stats if available
        if self.distributed_manager:
            status["distributed_info"] = {
                "is_distributed": self.distributed_manager.is_distributed,
                "rank": self.distributed_manager.rank,
                "world_size": self.distributed_manager.world_size,
                "backend": self.distributed_manager.backend
            }
        
        if self.load_balancer:
            status["load_balancer_stats"] = self.load_balancer.get_stats()
        
        return status

__all__ = [
    "DistributedTrainingManager",
    "AsyncInferenceEngine", 
    "ModelSharding",
    "LoadBalancer",
    "ScalingManager"
]