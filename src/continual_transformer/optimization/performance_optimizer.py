"""Performance optimization modules for continual learning transformers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import time
import psutil
import gc
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    inference_time: float
    memory_usage: float
    throughput: float
    accuracy: float
    efficiency_score: float
    energy_consumption: Optional[float] = None


class PerformanceOptimizer:
    """Comprehensive performance optimization for continual learning models."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.metrics_history = []
        self.optimization_strategies = []
        
        # Performance tracking
        self.start_time = None
        self.memory_baseline = None
        
    def optimize_inference(self, enable_optimizations: List[str] = None) -> Dict[str, Any]:
        """Apply inference optimizations."""
        if enable_optimizations is None:
            enable_optimizations = ["torch_compile", "quantization", "pruning", "fusion"]
        
        optimizations_applied = {}
        
        # Torch 2.0 compilation for speed
        if "torch_compile" in enable_optimizations:
            try:
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode="max-autotune")
                    optimizations_applied["torch_compile"] = True
                    logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"Torch compile failed: {e}")
                optimizations_applied["torch_compile"] = False
        
        # Dynamic quantization
        if "quantization" in enable_optimizations:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, 
                    {nn.Linear}, 
                    dtype=torch.qint8
                )
                optimizations_applied["quantization"] = True
                logger.info("Applied dynamic quantization")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
                optimizations_applied["quantization"] = False
        
        # Structured pruning for adapter layers
        if "pruning" in enable_optimizations:
            try:
                pruning_ratio = self.apply_structured_pruning()
                optimizations_applied["pruning"] = pruning_ratio
                logger.info(f"Applied structured pruning: {pruning_ratio:.2%} parameters removed")
            except Exception as e:
                logger.warning(f"Pruning failed: {e}")
                optimizations_applied["pruning"] = False
        
        # Operator fusion
        if "fusion" in enable_optimizations:
            try:
                self.apply_operator_fusion()
                optimizations_applied["fusion"] = True
                logger.info("Applied operator fusion")
            except Exception as e:
                logger.warning(f"Operator fusion failed: {e}")
                optimizations_applied["fusion"] = False
        
        return optimizations_applied
    
    def apply_structured_pruning(self, target_sparsity: float = 0.1) -> float:
        """Apply structured pruning to adapter layers."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if "adapter" in name.lower() and isinstance(module, nn.Linear):
                # Calculate importance scores (L2 norm of weights)
                weight = module.weight.data
                importance_scores = torch.norm(weight, dim=1)
                
                # Determine pruning threshold
                num_filters = weight.size(0)
                num_to_prune = int(num_filters * target_sparsity)
                
                if num_to_prune > 0:
                    # Get indices of least important filters
                    _, indices_to_prune = torch.topk(
                        importance_scores, 
                        num_to_prune, 
                        largest=False
                    )
                    
                    # Zero out the weights (structured pruning)
                    with torch.no_grad():
                        weight[indices_to_prune] = 0
                        if module.bias is not None:
                            module.bias[indices_to_prune] = 0
                    
                    pruned_params += num_to_prune * weight.size(1)
                
                total_params += weight.numel()
        
        return pruned_params / total_params if total_params > 0 else 0.0
    
    def apply_operator_fusion(self):
        """Apply operator fusion optimizations."""
        # This is a placeholder for operator fusion
        # In practice, this would involve graph-level optimizations
        # For now, we ensure the model is in the right mode
        self.model.eval()
        
        # Fuse batch norm and conv layers if present
        for module in self.model.modules():
            if hasattr(module, 'fuse_model'):
                module.fuse_model()
    
    def benchmark_performance(
        self, 
        test_data: torch.Tensor, 
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> PerformanceMetrics:
        """Comprehensive performance benchmarking."""
        
        # Warmup runs
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(test_data)
        
        # Clear cache and measure baseline memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        memory_before = self.get_memory_usage()
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                outputs = self.model(test_data)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        memory_after = self.get_memory_usage()
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = test_data.size(0) / avg_time  # samples per second
        memory_usage = memory_after - memory_before
        
        # Efficiency score (throughput / memory_usage)
        efficiency_score = throughput / max(memory_usage, 1e-6)
        
        metrics = PerformanceMetrics(
            inference_time=avg_time,
            memory_usage=memory_usage,
            throughput=throughput,
            accuracy=0.0,  # To be filled by evaluation
            efficiency_score=efficiency_score
        )
        
        self.metrics_history.append(metrics)
        
        logger.info(
            f"Performance Benchmark Results:\n"
            f"  Average inference time: {avg_time:.4f}s (±{std_time:.4f}s)\n"
            f"  Throughput: {throughput:.2f} samples/sec\n"
            f"  Memory usage: {memory_usage:.2f} MB\n"
            f"  Efficiency score: {efficiency_score:.2f}"
        )
        
        return metrics
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Apply memory optimization strategies."""
        optimizations = {}
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            optimizations['gradient_checkpointing'] = True
        
        # Clear unnecessary caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        optimizations['cache_cleared'] = True
        
        # Model parameter sharing for similar adapters
        shared_params = self.share_adapter_parameters()
        optimizations['shared_parameters'] = shared_params
        
        return optimizations
    
    def share_adapter_parameters(self) -> int:
        """Share parameters between similar adapters to reduce memory."""
        shared_count = 0
        
        # Get all adapter modules
        adapters = {}
        for name, module in self.model.named_modules():
            if "adapter" in name.lower():
                adapters[name] = module
        
        # Simple parameter sharing based on similar weight norms
        adapter_names = list(adapters.keys())
        for i in range(len(adapter_names)):
            for j in range(i + 1, len(adapter_names)):
                adapter1 = adapters[adapter_names[i]]
                adapter2 = adapters[adapter_names[j]]
                
                if self.adapters_are_similar(adapter1, adapter2):
                    # Share parameters (this is a simplified example)
                    # In practice, you'd need more sophisticated sharing logic
                    shared_count += 1
        
        return shared_count
    
    def adapters_are_similar(self, adapter1: nn.Module, adapter2: nn.Module) -> bool:
        """Check if two adapters are similar enough to share parameters."""
        # Simple similarity check based on weight norms
        try:
            norm1 = sum(torch.norm(p).item() for p in adapter1.parameters())
            norm2 = sum(torch.norm(p).item() for p in adapter2.parameters())
            
            similarity = abs(norm1 - norm2) / max(norm1, norm2, 1e-6)
            return similarity < 0.1  # 10% threshold
        except:
            return False


class MemoryOptimizer:
    """Specialized memory optimization for continual learning."""
    
    def __init__(self, model):
        self.model = model
        self.memory_snapshots = []
    
    def optimize_for_inference(self) -> Dict[str, Any]:
        """Optimize model specifically for inference."""
        optimizations = {}
        
        # Convert to half precision if supported
        if torch.cuda.is_available():
            try:
                self.model = self.model.half()
                optimizations['half_precision'] = True
            except:
                optimizations['half_precision'] = False
        
        # Freeze all parameters to save memory
        for param in self.model.parameters():
            param.requires_grad = False
        optimizations['parameters_frozen'] = True
        
        # Enable CUDNN benchmark mode
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            optimizations['cudnn_benchmark'] = True
        
        return optimizations
    
    def create_memory_snapshot(self) -> Dict[str, float]:
        """Create a snapshot of current memory usage."""
        snapshot = {
            'timestamp': time.time(),
            'gpu_memory': 0.0,
            'cpu_memory': 0.0,
            'model_parameters': 0.0
        }
        
        if torch.cuda.is_available():
            snapshot['gpu_memory'] = torch.cuda.memory_allocated() / 1024 / 1024
        
        process = psutil.Process()
        snapshot['cpu_memory'] = process.memory_info().rss / 1024 / 1024
        
        snapshot['model_parameters'] = sum(
            p.numel() * p.element_size() 
            for p in self.model.parameters()
        ) / 1024 / 1024
        
        self.memory_snapshots.append(snapshot)
        return snapshot


class ComputeOptimizer:
    """Specialized compute optimization for continual learning."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def optimize_compute_graph(self) -> Dict[str, Any]:
        """Optimize the compute graph for efficiency."""
        optimizations = {}
        
        # Enable TensorFloat-32 on Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimizations['tf32_enabled'] = True
        
        # Optimize attention computation
        optimizations['attention_optimized'] = self.optimize_attention()
        
        # Enable flash attention if available
        optimizations['flash_attention'] = self.enable_flash_attention()
        
        return optimizations
    
    def optimize_attention(self) -> bool:
        """Optimize attention mechanisms in the model."""
        try:
            # Enable SDPA (Scaled Dot Product Attention) optimization
            for module in self.model.modules():
                if hasattr(module, 'enable_flash_attention'):
                    module.enable_flash_attention()
            return True
        except:
            return False
    
    def enable_flash_attention(self) -> bool:
        """Enable Flash Attention if available."""
        try:
            # Check if flash attention is available
            import flash_attn
            return True
        except ImportError:
            return False


class AdaptiveOptimizer:
    """Adaptive optimization that learns optimal settings."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimization_history = []
        self.current_strategy = {}
    
    def adaptive_optimize(self, performance_target: float = 0.9) -> Dict[str, Any]:
        """Adaptively optimize based on performance targets."""
        
        # Start with baseline measurements
        baseline_metrics = self.measure_baseline_performance()
        
        # Try different optimization strategies
        strategies = [
            self.strategy_speed_focused,
            self.strategy_memory_focused,
            self.strategy_balanced,
            self.strategy_accuracy_focused
        ]
        
        best_strategy = None
        best_score = -float('inf')
        
        for strategy in strategies:
            try:
                # Apply strategy
                optimizations = strategy()
                
                # Measure performance
                metrics = self.measure_performance()
                
                # Calculate composite score
                score = self.calculate_composite_score(metrics, performance_target)
                
                if score > best_score:
                    best_score = score
                    best_strategy = optimizations
                
                # Record results
                self.optimization_history.append({
                    'strategy': strategy.__name__,
                    'optimizations': optimizations,
                    'metrics': metrics,
                    'score': score
                })
                
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed: {e}")
        
        self.current_strategy = best_strategy or {}
        return self.current_strategy
    
    def strategy_speed_focused(self) -> Dict[str, Any]:
        """Speed-focused optimization strategy."""
        optimizer = PerformanceOptimizer(self.model, self.config)
        return optimizer.optimize_inference(['torch_compile', 'fusion'])
    
    def strategy_memory_focused(self) -> Dict[str, Any]:
        """Memory-focused optimization strategy."""
        memory_opt = MemoryOptimizer(self.model)
        return memory_opt.optimize_for_inference()
    
    def strategy_balanced(self) -> Dict[str, Any]:
        """Balanced optimization strategy."""
        perf_opt = PerformanceOptimizer(self.model, self.config)
        memory_opt = MemoryOptimizer(self.model)
        
        optimizations = {}
        optimizations.update(perf_opt.optimize_inference(['quantization', 'pruning']))
        optimizations.update(memory_opt.optimize_for_inference())
        
        return optimizations
    
    def strategy_accuracy_focused(self) -> Dict[str, Any]:
        """Accuracy-focused optimization strategy (minimal optimizations)."""
        return {'accuracy_focused': True}
    
    def measure_baseline_performance(self) -> PerformanceMetrics:
        """Measure baseline performance before optimization."""
        # This would measure actual performance metrics
        # For now, return dummy metrics
        return PerformanceMetrics(
            inference_time=1.0,
            memory_usage=100.0,
            throughput=10.0,
            accuracy=0.85,
            efficiency_score=0.1
        )
    
    def measure_performance(self) -> PerformanceMetrics:
        """Measure current performance metrics."""
        # This would measure actual performance metrics
        # For now, return dummy metrics
        return PerformanceMetrics(
            inference_time=0.8,
            memory_usage=80.0,
            throughput=12.0,
            accuracy=0.83,
            efficiency_score=0.15
        )
    
    def calculate_composite_score(self, metrics: PerformanceMetrics, target_accuracy: float) -> float:
        """Calculate a composite performance score."""
        # Weighted combination of metrics
        if metrics.accuracy < target_accuracy:
            return -1.0  # Penalize accuracy drops below target
        
        # Normalize and combine metrics
        speed_score = 1.0 / metrics.inference_time
        memory_score = 1.0 / metrics.memory_usage
        throughput_score = metrics.throughput
        accuracy_score = metrics.accuracy
        
        # Weighted combination
        composite_score = (
            0.3 * speed_score +
            0.2 * memory_score +
            0.3 * throughput_score +
            0.2 * accuracy_score
        )
        
        return composite_score