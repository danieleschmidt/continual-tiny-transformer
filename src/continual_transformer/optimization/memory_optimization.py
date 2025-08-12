"""
Advanced memory optimization for continual learning at scale.
Implements gradient checkpointing, memory mapping, and efficient caching.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
import logging
import gc
import psutil
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Iterator
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import weakref
import mmap
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    gpu_memory_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_available_mb: float = 0.0
    timestamp: float = 0.0


class MemoryMonitor:
    """Real-time memory monitoring and alerting."""
    
    def __init__(self, alert_threshold: float = 0.85, check_interval: float = 1.0):
        self.alert_threshold = alert_threshold
        self.check_interval = check_interval
        self.monitoring_active = False
        self.memory_history = []
        self.alert_callbacks = []
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.check_interval):
            try:
                stats = self.get_memory_stats()
                self.memory_history.append(stats)
                
                # Keep only recent history
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-500:]
                
                # Check for alerts
                self._check_alerts(stats)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()
        
        stats = MemoryStats(
            total_memory_mb=memory.total / 1024 / 1024,
            used_memory_mb=memory.used / 1024 / 1024,
            available_memory_mb=memory.available / 1024 / 1024,
            timestamp=time.time()
        )
        
        # GPU memory if available
        if torch.cuda.is_available():
            try:
                stats.gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                stats.gpu_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
                stats.gpu_available_mb = stats.gpu_memory_mb - stats.gpu_used_mb
            except Exception as e:
                logger.debug(f"GPU memory check failed: {e}")
        
        return stats
    
    def _check_alerts(self, stats: MemoryStats):
        """Check if memory usage triggers alerts."""
        # System memory alert
        if stats.used_memory_mb / stats.total_memory_mb > self.alert_threshold:
            self._trigger_alert("system_memory", stats)
        
        # GPU memory alert
        if stats.gpu_memory_mb > 0:
            if stats.gpu_used_mb / stats.gpu_memory_mb > self.alert_threshold:
                self._trigger_alert("gpu_memory", stats)
    
    def _trigger_alert(self, alert_type: str, stats: MemoryStats):
        """Trigger memory alert."""
        logger.warning(f"Memory alert triggered: {alert_type}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, stats)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback: Callable[[str, MemoryStats], None]):
        """Register callback for memory alerts."""
        self.alert_callbacks.append(callback)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_history:
            return {"status": "no_data"}
        
        recent_stats = self.memory_history[-10:]
        current = self.memory_history[-1]
        
        return {
            "current": {
                "system_usage_pct": current.used_memory_mb / current.total_memory_mb * 100,
                "gpu_usage_pct": current.gpu_used_mb / max(current.gpu_memory_mb, 1) * 100,
                "available_mb": current.available_memory_mb
            },
            "trend": {
                "avg_system_usage": np.mean([s.used_memory_mb / s.total_memory_mb for s in recent_stats]) * 100,
                "avg_gpu_usage": np.mean([s.gpu_used_mb / max(s.gpu_memory_mb, 1) for s in recent_stats]) * 100
            },
            "monitoring_active": self.monitoring_active
        }


class GradientCheckpointing:
    """Advanced gradient checkpointing for memory-efficient training."""
    
    def __init__(self, model, checkpoint_segments: int = 4):
        self.model = model
        self.checkpoint_segments = checkpoint_segments
        self.checkpointed_modules = []
        self.original_forwards = {}
    
    def enable_checkpointing(self, modules: Optional[List[nn.Module]] = None):
        """Enable gradient checkpointing for specified modules."""
        
        if modules is None:
            # Auto-detect modules to checkpoint
            modules = self._auto_detect_checkpoint_modules()
        
        for module in modules:
            if module not in self.checkpointed_modules:
                self._wrap_module_with_checkpointing(module)
                self.checkpointed_modules.append(module)
        
        logger.info(f"Enabled gradient checkpointing for {len(self.checkpointed_modules)} modules")
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        
        for module in self.checkpointed_modules:
            self._unwrap_module(module)
        
        self.checkpointed_modules.clear()
        self.original_forwards.clear()
        
        logger.info("Disabled gradient checkpointing")
    
    def _auto_detect_checkpoint_modules(self) -> List[nn.Module]:
        """Auto-detect modules suitable for checkpointing."""
        checkpoint_modules = []
        
        # Look for transformer layers, attention modules, etc.
        for name, module in self.model.named_modules():
            # Common patterns for checkpointing
            if any(pattern in name.lower() for pattern in ['layer', 'block', 'encoder', 'decoder']):
                if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                    checkpoint_modules.append(module)
                elif hasattr(module, 'forward') and len(list(module.children())) > 0:
                    checkpoint_modules.append(module)
        
        return checkpoint_modules
    
    def _wrap_module_with_checkpointing(self, module: nn.Module):
        """Wrap module's forward method with checkpointing."""
        
        if hasattr(module, '_original_forward'):
            return  # Already wrapped
        
        original_forward = module.forward
        module._original_forward = original_forward
        
        def checkpointed_forward(*args, **kwargs):
            if self.model.training:
                return checkpoint.checkpoint(original_forward, *args, **kwargs)
            else:
                return original_forward(*args, **kwargs)
        
        module.forward = checkpointed_forward
    
    def _unwrap_module(self, module: nn.Module):
        """Unwrap module's forward method."""
        
        if hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            delattr(module, '_original_forward')


class MemoryMappedCache:
    """Memory-mapped cache for efficient data storage and retrieval."""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        self.cache_index = {}
        self.access_times = {}
        self.current_size = 0
        self._lock = threading.Lock()
        
        self._load_existing_cache()
    
    def _load_existing_cache(self):
        """Load existing cache files and build index."""
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                key = cache_file.stem
                size = cache_file.stat().st_size
                self.cache_index[key] = cache_file
                self.current_size += size
                self.access_times[key] = cache_file.stat().st_atime
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
    
    def put(self, key: str, data: Any) -> bool:
        """Store data in memory-mapped cache."""
        
        with self._lock:
            try:
                # Serialize data
                serialized_data = pickle.dumps(data)
                data_size = len(serialized_data)
                
                # Check if we need to evict items
                self._ensure_space(data_size)
                
                # Write to cache file
                cache_file = self.cache_dir / f"{key}.cache"
                with open(cache_file, 'wb') as f:
                    f.write(serialized_data)
                
                # Update index
                self.cache_index[key] = cache_file
                self.access_times[key] = time.time()
                self.current_size += data_size
                
                logger.debug(f"Cached {key} ({data_size} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache {key}: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from memory-mapped cache."""
        
        with self._lock:
            if key not in self.cache_index:
                return None
            
            try:
                cache_file = self.cache_index[key]
                
                # Memory-map the file for efficient reading
                with open(cache_file, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        data = pickle.loads(mm[:])
                
                # Update access time
                self.access_times[key] = time.time()
                
                logger.debug(f"Retrieved {key} from cache")
                return data
                
            except Exception as e:
                logger.error(f"Failed to retrieve {key}: {e}")
                # Remove corrupted cache entry
                self._remove_cache_entry(key)
                return None
    
    def _ensure_space(self, required_size: int):
        """Ensure there's enough space in cache by evicting old entries."""
        
        while self.current_size + required_size > self.max_size_bytes:
            if not self.cache_index:
                break
            
            # Find least recently used item
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._remove_cache_entry(oldest_key)
    
    def _remove_cache_entry(self, key: str):
        """Remove cache entry."""
        
        if key in self.cache_index:
            try:
                cache_file = self.cache_index[key]
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                
                del self.cache_index[key]
                del self.access_times[key]
                self.current_size -= file_size
                
                logger.debug(f"Evicted {key} from cache")
                
            except Exception as e:
                logger.error(f"Failed to remove cache entry {key}: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        
        with self._lock:
            for cache_file in self.cache_index.values():
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove {cache_file}: {e}")
            
            self.cache_index.clear()
            self.access_times.clear()
            self.current_size = 0
            
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        with self._lock:
            return {
                "total_entries": len(self.cache_index),
                "total_size_mb": self.current_size / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "usage_pct": self.current_size / self.max_size_bytes * 100,
                "cache_dir": str(self.cache_dir)
            }


class MemoryOptimizer:
    """Main memory optimization system."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Initialize components
        self.memory_monitor = MemoryMonitor(
            alert_threshold=getattr(config, 'memory_alert_threshold', 0.85)
        )
        
        self.gradient_checkpointing = GradientCheckpointing(
            model, 
            checkpoint_segments=getattr(config, 'checkpoint_segments', 4)
        )
        
        cache_dir = getattr(config, 'cache_dir', './cache')
        cache_size = getattr(config, 'cache_size_gb', 5.0)
        self.cache = MemoryMappedCache(cache_dir, cache_size)
        
        # Optimization state
        self.optimizations_active = {}
        self.memory_pressure_threshold = 0.8
        
        # Register memory alert callback
        self.memory_monitor.register_alert_callback(self._handle_memory_alert)
        
        logger.info("Memory optimization system initialized")
    
    def start_optimization(self):
        """Start memory optimization services."""
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Enable basic optimizations
        self._enable_basic_optimizations()
        
        logger.info("Memory optimization started")
    
    def stop_optimization(self):
        """Stop memory optimization services."""
        
        # Stop monitoring
        self.memory_monitor.stop_monitoring()
        
        # Disable optimizations
        self._disable_all_optimizations()
        
        logger.info("Memory optimization stopped")
    
    def _enable_basic_optimizations(self):
        """Enable basic memory optimizations."""
        
        # Enable gradient checkpointing for training
        if self.model.training:
            self.gradient_checkpointing.enable_checkpointing()
            self.optimizations_active['gradient_checkpointing'] = True
        
        # Enable automatic garbage collection
        self._enable_automatic_gc()
        self.optimizations_active['automatic_gc'] = True
        
        # Optimize PyTorch settings
        self._optimize_pytorch_settings()
        self.optimizations_active['pytorch_optimization'] = True
    
    def _disable_all_optimizations(self):
        """Disable all active optimizations."""
        
        if self.optimizations_active.get('gradient_checkpointing', False):
            self.gradient_checkpointing.disable_checkpointing()
        
        self.optimizations_active.clear()
    
    def _handle_memory_alert(self, alert_type: str, stats: MemoryStats):
        """Handle memory pressure alerts."""
        
        logger.warning(f"Memory alert: {alert_type} - {stats.used_memory_mb:.1f}MB used")
        
        if alert_type == "system_memory":
            self._handle_system_memory_pressure(stats)
        elif alert_type == "gpu_memory":
            self._handle_gpu_memory_pressure(stats)
    
    def _handle_system_memory_pressure(self, stats: MemoryStats):
        """Handle system memory pressure."""
        
        # Clear cache partially
        self._partial_cache_clear(0.3)  # Clear 30% of cache
        
        # Force garbage collection
        self._force_garbage_collection()
        
        # Reduce batch sizes if possible
        self._suggest_batch_size_reduction()
    
    def _handle_gpu_memory_pressure(self, stats: MemoryStats):
        """Handle GPU memory pressure."""
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enable gradient checkpointing if not already active
        if not self.optimizations_active.get('gradient_checkpointing', False):
            self.gradient_checkpointing.enable_checkpointing()
            self.optimizations_active['gradient_checkpointing'] = True
        
        # Suggest model optimization
        self._suggest_model_optimization()
    
    def _partial_cache_clear(self, fraction: float):
        """Clear a fraction of the cache."""
        
        stats = self.cache.get_cache_stats()
        if stats['total_entries'] == 0:
            return
        
        entries_to_remove = int(stats['total_entries'] * fraction)
        
        # Remove oldest entries
        with self.cache._lock:
            oldest_keys = sorted(
                self.cache.access_times.keys(),
                key=lambda k: self.cache.access_times[k]
            )[:entries_to_remove]
            
            for key in oldest_keys:
                self.cache._remove_cache_entry(key)
        
        logger.info(f"Cleared {entries_to_remove} cache entries due to memory pressure")
    
    def _force_garbage_collection(self):
        """Force comprehensive garbage collection."""
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Forced garbage collection completed")
    
    def _suggest_batch_size_reduction(self):
        """Suggest batch size reduction."""
        
        if hasattr(self.config, 'batch_size') and self.config.batch_size > 1:
            suggested_size = max(1, self.config.batch_size // 2)
            logger.warning(
                f"Consider reducing batch size from {self.config.batch_size} to {suggested_size} "
                "due to memory pressure"
            )
    
    def _suggest_model_optimization(self):
        """Suggest model optimization techniques."""
        
        suggestions = []
        
        if not self.optimizations_active.get('gradient_checkpointing', False):
            suggestions.append("Enable gradient checkpointing")
        
        if hasattr(self.config, 'mixed_precision') and not self.config.mixed_precision:
            suggestions.append("Enable mixed precision training")
        
        if suggestions:
            logger.warning(f"Memory optimization suggestions: {', '.join(suggestions)}")
    
    def _enable_automatic_gc(self):
        """Enable automatic garbage collection optimization."""
        
        # Set more aggressive garbage collection
        gc.set_threshold(500, 5, 5)  # More frequent collection
    
    def _optimize_pytorch_settings(self):
        """Optimize PyTorch memory settings."""
        
        # Optimize CUDA memory allocation
        if torch.cuda.is_available():
            # Use more efficient memory allocation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction if needed
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                memory_fraction = getattr(self.config, 'gpu_memory_fraction', 0.9)
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    def optimize_model_for_inference(self):
        """Optimize model specifically for inference."""
        
        if not self.model.training:
            # Disable gradient computation globally
            torch.set_grad_enabled(False)
            
            # Optimize for evaluation
            self.model.eval()
            
            # Enable inference optimizations
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
            
            logger.info("Model optimized for inference")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        
        status = {
            "active_optimizations": list(self.optimizations_active.keys()),
            "memory_monitor": self.memory_monitor.get_memory_summary(),
            "cache_stats": self.cache.get_cache_stats(),
            "gradient_checkpointing": {
                "enabled": len(self.gradient_checkpointing.checkpointed_modules) > 0,
                "modules_count": len(self.gradient_checkpointing.checkpointed_modules)
            }
        }
        
        # Add current memory usage
        current_stats = self.memory_monitor.get_memory_stats()
        status["current_memory"] = {
            "system_usage_mb": current_stats.used_memory_mb,
            "gpu_usage_mb": current_stats.gpu_used_mb,
            "system_available_mb": current_stats.available_memory_mb,
            "gpu_available_mb": current_stats.gpu_available_mb
        }
        
        return status
    
    def cache_computation(self, key: str, computation_func: Callable, *args, **kwargs):
        """Cache expensive computations."""
        
        # Try to get from cache first
        result = self.cache.get(key)
        
        if result is not None:
            logger.debug(f"Cache hit for {key}")
            return result
        
        # Compute and cache result
        logger.debug(f"Cache miss for {key}, computing...")
        result = computation_func(*args, **kwargs)
        
        # Store in cache
        self.cache.put(key, result)
        
        return result