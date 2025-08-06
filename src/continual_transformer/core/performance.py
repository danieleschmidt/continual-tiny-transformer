"""Performance optimization and scaling features for continual transformers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
import psutil
import logging
from collections import defaultdict, deque
from pathlib import Path
import pickle
import json
import hashlib

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Advanced performance optimization for continual learning."""
    
    def __init__(self, config=None):
        self.config = config
        self.device = torch.device(config.device if config else "cpu")
        
        # Performance caches
        self.prediction_cache = {}
        self.embedding_cache = {}
        self.attention_cache = {}
        self.max_cache_size = 10000
        
        # Performance metrics
        self.timing_stats = defaultdict(list)
        self.memory_stats = deque(maxlen=1000)
        self.throughput_stats = deque(maxlen=100)
        
        # Optimization settings
        self.enable_mixed_precision = getattr(config, 'mixed_precision', True) if config else True
        self.enable_gradient_checkpointing = getattr(config, 'gradient_checkpointing', False) if config else False
        self.enable_compilation = True  # PyTorch 2.0+ compilation
        
        # Threading and async settings
        self.thread_pool = None
        self.max_workers = min(4, multiprocessing.cpu_count())
        
        logger.info(f"Performance optimizer initialized for device: {self.device}")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive model optimizations."""
        start_time = time.time()
        
        try:
            # 1. Enable mixed precision if supported
            if self.enable_mixed_precision and self.device.type == 'cuda':
                model = self._apply_mixed_precision(model)
            
            # 2. Enable gradient checkpointing for memory efficiency
            if self.enable_gradient_checkpointing:
                model = self._apply_gradient_checkpointing(model)
            
            # 3. Optimize for inference
            model = self._optimize_for_inference(model)
            
            # 4. Apply PyTorch 2.0+ compilation if available
            if self.enable_compilation:
                model = self._apply_compilation(model)
            
            # 5. Memory optimization
            self._optimize_memory_layout(model)
            
            optimization_time = time.time() - start_time
            self.timing_stats['model_optimization'].append(optimization_time)
            
            logger.info(f"Model optimization completed in {optimization_time:.3f}s")
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model  # Return original model if optimization fails
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Apply automatic mixed precision optimization."""
        try:
            from torch.cuda.amp import autocast
            
            class MixedPrecisionWrapper(nn.Module):
                def __init__(self, wrapped_model):
                    super().__init__()
                    self.model = wrapped_model
                    
                def forward(self, *args, **kwargs):
                    with autocast():
                        return self.model(*args, **kwargs)
            
            logger.info("Applied mixed precision optimization")
            return MixedPrecisionWrapper(model)
            
        except ImportError:
            logger.warning("Mixed precision not available, skipping")
            return model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing for memory efficiency."""
        try:
            # Enable gradient checkpointing for transformer layers
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'encoder'):
                for layer in model.base_model.encoder.layer:
                    if hasattr(layer, 'gradient_checkpointing'):
                        layer.gradient_checkpointing = True
            
            logger.info("Applied gradient checkpointing")
            
        except Exception as e:
            logger.warning(f"Gradient checkpointing failed: {e}")
        
        return model
    
    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference speed."""
        try:
            # Fuse operations where possible
            model = torch.jit.optimize_for_inference(model)
            logger.info("Applied inference optimizations")
            
        except Exception as e:
            logger.warning(f"Inference optimization failed: {e}")
        
        return model
    
    def _apply_compilation(self, model: nn.Module) -> nn.Module:
        """Apply PyTorch 2.0+ compilation."""
        try:
            if hasattr(torch, 'compile'):
                # Apply different compilation modes based on use case
                if self.device.type == 'cuda':
                    model = torch.compile(model, mode="reduce-overhead")
                else:
                    model = torch.compile(model, mode="default")
                logger.info("Applied PyTorch compilation")
                
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def _optimize_memory_layout(self, model: nn.Module):
        """Optimize memory layout for better cache efficiency."""
        try:
            # Convert to channels_last memory format if on GPU
            if self.device.type == 'cuda':
                for module in model.modules():
                    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                        module = module.to(memory_format=torch.channels_last)
            
            logger.info("Applied memory layout optimizations")
            
        except Exception as e:
            logger.warning(f"Memory layout optimization failed: {e}")
    
    @lru_cache(maxsize=1000)
    def cached_embedding_lookup(self, input_hash: str, model_embeddings) -> torch.Tensor:
        """Cached embedding lookup for repeated inputs."""
        # This would be implemented with actual embedding computation
        # For now, return a placeholder
        return None
    
    def batch_inference(
        self, 
        model: nn.Module, 
        inputs: List[Dict[str, torch.Tensor]], 
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """Optimized batch inference with dynamic batching."""
        results = []
        
        # Sort inputs by length for better batching efficiency
        sorted_inputs = sorted(inputs, key=lambda x: x['input_ids'].size(-1))
        
        for i in range(0, len(sorted_inputs), batch_size):
            batch_inputs = sorted_inputs[i:i+batch_size]
            
            # Create padded batch
            batch = self._create_padded_batch(batch_inputs)
            
            # Perform inference
            with torch.no_grad():
                start_time = time.time()
                batch_results = model(**batch)
                inference_time = time.time() - start_time
                
                # Record performance metrics
                self.timing_stats['inference'].append(inference_time)
                self.throughput_stats.append(len(batch_inputs) / inference_time)
            
            # Split batch results back to individual results
            individual_results = self._split_batch_results(batch_results, len(batch_inputs))
            results.extend(individual_results)
        
        return results
    
    def _create_padded_batch(self, inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Create padded batch from individual inputs."""
        if not inputs:
            return {}
        
        # Find max length
        max_length = max(inp['input_ids'].size(-1) for inp in inputs)
        batch_size = len(inputs)
        
        # Create padded tensors
        padded_batch = {}
        for key in inputs[0].keys():
            if key in ['input_ids', 'attention_mask']:
                # Pad sequences
                padded_tensor = torch.zeros(batch_size, max_length, dtype=inputs[0][key].dtype)
                for i, inp in enumerate(inputs):
                    seq_len = inp[key].size(-1)
                    padded_tensor[i, :seq_len] = inp[key].squeeze(0)
                padded_batch[key] = padded_tensor.to(self.device)
            else:
                # Stack non-sequence data
                padded_batch[key] = torch.stack([inp[key] for inp in inputs]).to(self.device)
        
        return padded_batch
    
    def _split_batch_results(self, batch_results: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Split batch results back to individual results."""
        results = []
        for i in range(batch_size):
            result = {}
            for key, tensor in batch_results.items():
                if tensor.dim() > 0:  # Skip scalars
                    result[key] = tensor[i:i+1]  # Keep batch dimension
                else:
                    result[key] = tensor
            results.append(result)
        return results
    
    def parallel_task_processing(
        self,
        model: nn.Module,
        tasks: List[Tuple[str, Any]],
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process multiple tasks in parallel."""
        if not tasks:
            return {}
        
        max_workers = max_workers or self.max_workers
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(self._process_single_task, model, task_id, task_data): task_id
                for task_id, task_data in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    results[task_id] = result
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    results[task_id] = {"error": str(e)}
        
        return results
    
    def _process_single_task(self, model: nn.Module, task_id: str, task_data: Any) -> Dict[str, Any]:
        """Process a single task (placeholder implementation)."""
        start_time = time.time()
        
        # Placeholder task processing
        # In real implementation, this would perform actual task processing
        result = {
            "task_id": task_id,
            "status": "completed",
            "processing_time": time.time() - start_time
        }
        
        return result
    
    def memory_efficient_training(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4
    ) -> Dict[str, float]:
        """Memory-efficient training with gradient accumulation."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Clear gradients
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast() if self.device.type == 'cuda' else torch.no_grad():
                outputs = model(**batch)
                loss = outputs.get('loss', outputs.get('logits', torch.tensor(0.0)))
                
                if torch.is_tensor(loss) and loss.dim() > 0:
                    loss = loss.mean()
                
                # Scale loss for accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            if self.enable_mixed_precision and self.device.type == 'cuda':
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.enable_mixed_precision and self.device.type == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                num_batches += 1
            
            # Memory cleanup
            del outputs, loss
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return {
            "average_loss": total_loss / max(num_batches, 1),
            "total_batches": num_batches
        }
    
    def adaptive_batch_sizing(
        self,
        model: nn.Module,
        sample_batch: Dict[str, torch.Tensor],
        target_memory_usage: float = 0.8
    ) -> int:
        """Determine optimal batch size based on memory usage."""
        if self.device.type != 'cuda':
            return 16  # Default for CPU
        
        # Get available memory
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        target_memory = total_memory * target_memory_usage
        
        # Start with batch size of 1 and increase
        batch_size = 1
        max_batch_size = 256
        
        model.eval()
        with torch.no_grad():
            while batch_size <= max_batch_size:
                try:
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                    # Create test batch
                    test_batch = self._create_test_batch(sample_batch, batch_size)
                    
                    # Forward pass to measure memory
                    _ = model(**test_batch)
                    
                    # Check memory usage
                    current_memory = torch.cuda.memory_allocated(self.device)
                    
                    if current_memory > target_memory:
                        # Use previous batch size
                        optimal_batch_size = max(batch_size // 2, 1)
                        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
                        return optimal_batch_size
                    
                    batch_size *= 2
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        optimal_batch_size = max(batch_size // 2, 1)
                        logger.info(f"Optimal batch size determined (OOM): {optimal_batch_size}")
                        return optimal_batch_size
                    else:
                        raise e
        
        logger.info(f"Optimal batch size determined (max): {batch_size // 2}")
        return batch_size // 2
    
    def _create_test_batch(self, sample_batch: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
        """Create test batch for memory testing."""
        test_batch = {}
        for key, tensor in sample_batch.items():
            if torch.is_tensor(tensor):
                # Repeat tensor to create batch
                if tensor.dim() == 0:  # Scalar
                    test_batch[key] = tensor.repeat(batch_size)
                else:
                    test_batch[key] = tensor.repeat(batch_size, *([1] * (tensor.dim() - 1)))
                test_batch[key] = test_batch[key].to(self.device)
            else:
                test_batch[key] = tensor
        return test_batch
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        summary = {
            "timing_stats": {},
            "memory_stats": {
                "samples": len(self.memory_stats),
                "avg_usage_mb": np.mean(self.memory_stats) if self.memory_stats else 0
            },
            "throughput_stats": {
                "samples": len(self.throughput_stats),
                "avg_throughput": np.mean(self.throughput_stats) if self.throughput_stats else 0,
                "max_throughput": max(self.throughput_stats) if self.throughput_stats else 0
            },
            "cache_stats": {
                "prediction_cache_size": len(self.prediction_cache),
                "embedding_cache_size": len(self.embedding_cache),
                "attention_cache_size": len(self.attention_cache)
            }
        }
        
        # Process timing stats
        for operation, times in self.timing_stats.items():
            if times:
                summary["timing_stats"][operation] = {
                    "count": len(times),
                    "avg_time": np.mean(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }
        
        return summary
    
    def clear_caches(self):
        """Clear all performance caches."""
        self.prediction_cache.clear()
        self.embedding_cache.clear()
        self.attention_cache.clear()
        
        if hasattr(self.cached_embedding_lookup, 'cache_clear'):
            self.cached_embedding_lookup.cache_clear()
        
        logger.info("Performance caches cleared")
    
    def warmup_model(self, model: nn.Module, sample_inputs: List[Dict[str, torch.Tensor]], warmup_steps: int = 10):
        """Warmup model for optimal performance."""
        logger.info(f"Warming up model with {warmup_steps} steps")
        
        model.eval()
        with torch.no_grad():
            for step in range(warmup_steps):
                for sample_input in sample_inputs[:min(5, len(sample_inputs))]:  # Limit warmup inputs
                    try:
                        # Move to device
                        input_batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                                     for k, v in sample_input.items()}
                        
                        # Forward pass
                        _ = model(**input_batch)
                        
                    except Exception as e:
                        logger.warning(f"Warmup step {step} failed: {e}")
                        continue
        
        # Clear GPU cache after warmup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info("Model warmup completed")
    
    def profile_model(self, model: nn.Module, sample_input: Dict[str, torch.Tensor], num_runs: int = 100) -> Dict[str, Any]:
        """Profile model performance comprehensively."""
        logger.info(f"Profiling model performance with {num_runs} runs")
        
        model.eval()
        input_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in sample_input.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(**input_batch)
        
        # Profile inference
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for run in range(num_runs):
                # Clear memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(self.device)
                
                # Time inference
                start_time = time.time()
                _ = model(**input_batch)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Memory usage
                if self.device.type == 'cuda':
                    peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024**2)  # MB
                    memory_usage.append(peak_memory)
        
        # Analyze results
        profile_results = {
            "inference_time": {
                "avg_ms": np.mean(inference_times) * 1000,
                "min_ms": min(inference_times) * 1000,
                "max_ms": max(inference_times) * 1000,
                "std_ms": np.std(inference_times) * 1000,
                "p95_ms": np.percentile(inference_times, 95) * 1000,
                "p99_ms": np.percentile(inference_times, 99) * 1000
            },
            "memory_usage": {
                "avg_mb": np.mean(memory_usage) if memory_usage else 0,
                "max_mb": max(memory_usage) if memory_usage else 0,
                "min_mb": min(memory_usage) if memory_usage else 0
            },
            "throughput": {
                "avg_samples_per_second": 1 / np.mean(inference_times) if inference_times else 0,
                "max_samples_per_second": 1 / min(inference_times) if inference_times else 0
            },
            "num_runs": num_runs,
            "device": str(self.device)
        }
        
        logger.info(f"Profiling completed: {profile_results['throughput']['avg_samples_per_second']:.2f} samples/sec")
        return profile_results
    
    def __del__(self):
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class ModelScaler:
    """Dynamic model scaling for varying workloads."""
    
    def __init__(self, config=None):
        self.config = config
        self.scaling_history = deque(maxlen=100)
        self.load_thresholds = {
            "scale_up": 0.8,    # Scale up when utilization > 80%
            "scale_down": 0.3   # Scale down when utilization < 30%
        }
        
    def should_scale(self, current_load: float, current_capacity: int) -> Tuple[bool, str, int]:
        """Determine if scaling is needed."""
        if current_load > self.load_thresholds["scale_up"]:
            new_capacity = min(current_capacity * 2, 16)  # Max 16 replicas
            return True, "up", new_capacity
        elif current_load < self.load_thresholds["scale_down"] and current_capacity > 1:
            new_capacity = max(current_capacity // 2, 1)  # Min 1 replica
            return True, "down", new_capacity
        
        return False, "none", current_capacity
    
    def adaptive_precision_scaling(self, model: nn.Module, target_latency: float) -> nn.Module:
        """Dynamically adjust model precision based on latency requirements."""
        # This would implement dynamic quantization/precision adjustment
        # For now, return the original model
        return model


# Utility functions for performance optimization
def benchmark_operation(func):
    """Decorator to benchmark function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f}s")
        return result
    
    return wrapper


def memory_efficient_forward(model: nn.Module, *args, **kwargs):
    """Memory-efficient forward pass with automatic cleanup."""
    try:
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            result = model(*args, **kwargs)
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CacheManager:
    """Intelligent caching system for model operations."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_count = defaultdict(int)
        self.access_time = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.access_count[key] += 1
            self.access_time[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with LRU eviction."""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_count[key] += 1
        self.access_time[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_time:
            return
        
        lru_key = min(self.access_time, key=self.access_time.get)
        del self.cache[lru_key]
        del self.access_count[lru_key]
        del self.access_time[lru_key]
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()
        self.access_time.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": sum(self.access_count.values()) / max(len(self.cache), 1),
            "total_accesses": sum(self.access_count.values())
        }