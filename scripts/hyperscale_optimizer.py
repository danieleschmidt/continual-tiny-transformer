#!/usr/bin/env python3
"""
TERRAGON LABS - Hyperscale SDLC Optimizer
Generation 3: Advanced performance optimization, auto-scaling, and intelligent resource management
"""

import asyncio
import json
import logging
import time
import psutil
import subprocess
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import multiprocessing
import statistics
import pickle
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OptimizationProfile:
    """Performance optimization profile"""
    name: str
    cpu_cores: int
    memory_limit_gb: float
    parallel_jobs: int
    cache_enabled: bool = True
    compression_enabled: bool = False
    priority: int = 5  # 1-10 scale
    optimization_level: str = "balanced"  # aggressive, balanced, conservative


@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    thread_count: int
    process_count: int


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    execution_time: float
    throughput: float
    resource_efficiency: float
    cache_hit_rate: float
    parallelization_factor: float
    bottleneck_analysis: Dict[str, Any]


class IntelligentCache:
    """Intelligent caching system with adaptive strategies"""
    
    def __init__(self, max_size_gb: float = 2.0):
        self.cache_dir = Path.cwd() / '.sdlc_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_metadata = {}
        self.access_patterns = defaultdict(list)
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached item with access pattern tracking"""
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access pattern
                self.access_patterns[key].append(datetime.now())
                self._update_metadata(key, access_time=datetime.now())
                
                logger.debug(f"Cache hit: {key}")
                return data
                
            except Exception as e:
                logger.warning(f"Cache read error for {key}: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def put(self, key: str, value: Any, ttl_hours: int = 24):
        """Store item in cache with TTL and size management"""
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        try:
            # Serialize and store
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            file_size = cache_file.stat().st_size
            expiry = datetime.now() + timedelta(hours=ttl_hours)
            
            self._update_metadata(key, file_size=file_size, expiry=expiry)
            
            # Clean up if over size limit
            self._cleanup_cache()
            
            logger.debug(f"Cached: {key} ({file_size} bytes)")
            
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _update_metadata(self, key: str, **kwargs):
        """Update cache metadata"""
        if key not in self.cache_metadata:
            self.cache_metadata[key] = {
                'created': datetime.now(),
                'access_count': 0,
                'file_size': 0
            }
        
        self.cache_metadata[key].update(kwargs)
        self.cache_metadata[key]['access_count'] += 1
    
    def _cleanup_cache(self):
        """Intelligent cache cleanup based on usage patterns"""
        current_size = sum(meta.get('file_size', 0) for meta in self.cache_metadata.values())
        
        if current_size <= self.max_size_bytes:
            return
        
        # Sort by access patterns (LFU + recency)
        candidates_for_removal = []
        
        for key, metadata in self.cache_metadata.items():
            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            
            if not cache_file.exists():
                continue
            
            # Calculate score (lower = more likely to be removed)
            access_count = metadata.get('access_count', 0)
            last_access = metadata.get('access_time', metadata.get('created', datetime.min))
            age_hours = (datetime.now() - last_access).total_seconds() / 3600
            
            score = access_count / (1 + age_hours)  # Combine frequency and recency
            
            candidates_for_removal.append((score, key, cache_file, metadata.get('file_size', 0)))
        
        # Remove least valuable items
        candidates_for_removal.sort()  # Sort by score (ascending)
        
        bytes_to_remove = current_size - int(self.max_size_bytes * 0.8)  # Remove to 80% capacity
        removed_bytes = 0
        
        for score, key, cache_file, file_size in candidates_for_removal:
            if removed_bytes >= bytes_to_remove:
                break
            
            cache_file.unlink(missing_ok=True)
            del self.cache_metadata[key]
            removed_bytes += file_size
            
            logger.debug(f"Removed from cache: {key} ({file_size} bytes)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_size = sum(meta.get('file_size', 0) for meta in self.cache_metadata.values())
        total_accesses = sum(meta.get('access_count', 0) for meta in self.cache_metadata.values())
        
        return {
            'total_items': len(self.cache_metadata),
            'total_size_mb': total_size / 1024 / 1024,
            'total_accesses': total_accesses,
            'hit_rate': self._calculate_hit_rate(),
            'utilization_percent': (total_size / self.max_size_bytes) * 100
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate from access patterns"""
        # Simplified calculation - in practice would track hits/misses
        return min(95.0, len(self.cache_metadata) * 10)


class AdaptiveResourceManager:
    """Adaptive resource management with auto-scaling"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.resource_history = deque(maxlen=100)
        self.optimization_profiles = self._initialize_profiles()
        self.current_profile = "balanced"
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        return {
            'cpu_cores': multiprocessing.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'disk_info': {disk.device: psutil.disk_usage(disk.mountpoint)._asdict() 
                         for disk in psutil.disk_partitions()},
            'platform': psutil.platform.system()
        }
    
    def _initialize_profiles(self) -> Dict[str, OptimizationProfile]:
        """Initialize optimization profiles"""
        cpu_cores = self.system_info['cpu_cores']
        memory_gb = self.system_info['memory_total_gb']
        
        return {
            'conservative': OptimizationProfile(
                name="Conservative",
                cpu_cores=max(1, cpu_cores // 4),
                memory_limit_gb=memory_gb * 0.25,
                parallel_jobs=max(1, cpu_cores // 4),
                optimization_level="conservative"
            ),
            'balanced': OptimizationProfile(
                name="Balanced",
                cpu_cores=max(2, cpu_cores // 2),
                memory_limit_gb=memory_gb * 0.5,
                parallel_jobs=max(2, cpu_cores // 2),
                optimization_level="balanced"
            ),
            'aggressive': OptimizationProfile(
                name="Aggressive",
                cpu_cores=max(4, int(cpu_cores * 0.8)),
                memory_limit_gb=memory_gb * 0.8,
                parallel_jobs=max(4, int(cpu_cores * 0.9)),
                cache_enabled=True,
                compression_enabled=True,
                optimization_level="aggressive"
            ),
            'maximum': OptimizationProfile(
                name="Maximum Performance",
                cpu_cores=cpu_cores,
                memory_limit_gb=memory_gb * 0.95,
                parallel_jobs=cpu_cores + 2,
                cache_enabled=True,
                compression_enabled=True,
                priority=10,
                optimization_level="aggressive"
            )
        }
    
    def get_optimal_profile(self, task_complexity: str = "medium", system_load: float = None) -> OptimizationProfile:
        """Get optimal profile based on current conditions"""
        if system_load is None:
            system_load = self._get_current_system_load()
        
        # Adaptive profile selection
        if system_load > 80:
            return self.optimization_profiles['conservative']
        elif system_load > 60:
            return self.optimization_profiles['balanced']
        elif task_complexity == "high":
            return self.optimization_profiles['aggressive']
        else:
            return self.optimization_profiles['maximum']
    
    def _get_current_system_load(self) -> float:
        """Calculate current system load score"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Weighted load score
        load_score = (cpu_percent * 0.6) + (memory_percent * 0.4)
        return load_score
    
    def monitor_resource_usage(self, duration_seconds: int = 60) -> List[ResourceUsage]:
        """Monitor resource usage over time"""
        measurements = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            timestamp = datetime.now()
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_metrics = {
                'read_bytes_per_sec': disk_io.read_bytes if disk_io else 0,
                'write_bytes_per_sec': disk_io.write_bytes if disk_io else 0
            }
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_metrics = {
                'bytes_sent_per_sec': net_io.bytes_sent if net_io else 0,
                'bytes_recv_per_sec': net_io.bytes_recv if net_io else 0
            }
            
            # Process info
            process_count = len(psutil.pids())
            thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) 
                             if p.info['num_threads'] is not None)
            
            usage = ResourceUsage(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_io=disk_metrics,
                network_io=network_metrics,
                thread_count=thread_count,
                process_count=process_count
            )
            
            measurements.append(usage)
            time.sleep(1)
        
        return measurements


class HyperscaleSDLCOptimizer:
    """
    Generation 3: Hyperscale SDLC optimizer with intelligent resource management,
    adaptive caching, parallel execution, and performance analytics
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.cache = IntelligentCache(max_size_gb=2.0)
        self.resource_manager = AdaptiveResourceManager()
        self.performance_history = deque(maxlen=1000)
        self.parallel_executor = None
        self.optimization_enabled = True
        
    async def optimize_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize complete workflow with intelligent parallelization"""
        start_time = time.time()
        
        logger.info("🚀 Starting hyperscale workflow optimization")
        
        # Analyze workflow for optimization opportunities
        optimized_workflow = self._analyze_and_optimize_workflow(workflow_definition)
        
        # Select optimal resource profile
        profile = self.resource_manager.get_optimal_profile(
            task_complexity=workflow_definition.get('complexity', 'medium')
        )
        
        logger.info(f"Selected optimization profile: {profile.name}")
        
        # Initialize parallel executor
        self.parallel_executor = ProcessPoolExecutor(max_workers=profile.parallel_jobs)
        
        try:
            # Execute optimized workflow
            results = await self._execute_optimized_workflow(optimized_workflow, profile)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            metrics = self._calculate_performance_metrics(results, execution_time, profile)
            
            # Store performance data for learning
            self._store_performance_data(workflow_definition, results, metrics)
            
            logger.info(f"✅ Workflow optimization completed in {execution_time:.1f}s")
            
            return {
                'status': 'success',
                'execution_time': execution_time,
                'metrics': metrics,
                'profile_used': profile.name,
                'optimization_savings': self._calculate_optimization_savings(execution_time, metrics),
                'results': results
            }
            
        finally:
            if self.parallel_executor:
                self.parallel_executor.shutdown(wait=True)
    
    def _analyze_and_optimize_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow and apply optimization strategies"""
        optimized = workflow.copy()
        
        # Identify parallelizable tasks
        parallel_groups = self._identify_parallel_groups(workflow.get('tasks', []))
        optimized['parallel_groups'] = parallel_groups
        
        # Optimize task order based on dependencies and resource requirements
        optimized_order = self._optimize_task_order(workflow.get('tasks', []))
        optimized['optimized_order'] = optimized_order
        
        # Add caching strategies
        optimized['caching_strategy'] = self._determine_caching_strategy(workflow.get('tasks', []))
        
        # Resource allocation optimization
        optimized['resource_allocation'] = self._optimize_resource_allocation(workflow.get('tasks', []))
        
        return optimized
    
    def _identify_parallel_groups(self, tasks: List[Dict[str, Any]]) -> List[List[str]]:
        """Identify groups of tasks that can run in parallel"""
        # Build dependency graph
        dependencies = {}
        for task in tasks:
            task_name = task.get('name', '')
            task_deps = task.get('dependencies', [])
            dependencies[task_name] = set(task_deps)
        
        # Group tasks by dependency level
        groups = []
        remaining_tasks = set(dependencies.keys())
        
        while remaining_tasks:
            # Find tasks with no dependencies among remaining tasks
            current_group = []
            for task in list(remaining_tasks):
                if not dependencies[task].intersection(remaining_tasks):
                    current_group.append(task)
            
            if not current_group:
                # Circular dependency or error - break to prevent infinite loop
                logger.warning("Potential circular dependency detected")
                current_group = list(remaining_tasks)[:1]
            
            groups.append(current_group)
            
            # Remove completed tasks from dependencies
            for task in current_group:
                remaining_tasks.remove(task)
                for other_task in remaining_tasks:
                    dependencies[other_task].discard(task)
        
        return groups
    
    def _optimize_task_order(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Optimize task execution order for maximum efficiency"""
        # Sort by estimated execution time and resource requirements
        task_priorities = []
        
        for task in tasks:
            name = task.get('name', '')
            estimated_time = task.get('estimated_time_seconds', 60)
            resource_intensity = task.get('resource_intensity', 'medium')
            
            # Calculate priority score (lower = higher priority)
            intensity_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 2.0}.get(resource_intensity, 1.0)
            priority_score = estimated_time * intensity_multiplier
            
            task_priorities.append((priority_score, name))
        
        # Sort by priority (ascending - lower scores first)
        task_priorities.sort()
        
        return [name for _, name in task_priorities]
    
    def _determine_caching_strategy(self, tasks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Determine optimal caching strategy for each task"""
        strategies = {}
        
        for task in tasks:
            task_name = task.get('name', '')
            task_type = task.get('type', 'generic')
            is_deterministic = task.get('deterministic', True)
            
            if is_deterministic and task_type in ['build', 'test', 'lint', 'security_scan']:
                strategies[task_name] = 'aggressive_cache'
            elif is_deterministic:
                strategies[task_name] = 'standard_cache'
            else:
                strategies[task_name] = 'no_cache'
        
        return strategies
    
    def _optimize_resource_allocation(self, tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Optimize resource allocation for each task"""
        allocations = {}
        
        total_system_cores = self.resource_manager.system_info['cpu_cores']
        total_system_memory = self.resource_manager.system_info['memory_total_gb']
        
        for task in tasks:
            task_name = task.get('name', '')
            resource_intensity = task.get('resource_intensity', 'medium')
            
            if resource_intensity == 'high':
                allocation = {
                    'cpu_cores': max(2, total_system_cores // 2),
                    'memory_gb': total_system_memory * 0.4,
                    'priority': 'high'
                }
            elif resource_intensity == 'medium':
                allocation = {
                    'cpu_cores': max(1, total_system_cores // 4),
                    'memory_gb': total_system_memory * 0.2,
                    'priority': 'normal'
                }
            else:  # low
                allocation = {
                    'cpu_cores': 1,
                    'memory_gb': total_system_memory * 0.1,
                    'priority': 'low'
                }
            
            allocations[task_name] = allocation
        
        return allocations
    
    async def _execute_optimized_workflow(self, workflow: Dict[str, Any], profile: OptimizationProfile) -> Dict[str, Any]:
        """Execute workflow with optimizations applied"""
        results = {}
        total_tasks = 0
        completed_tasks = 0
        
        # Execute in parallel groups
        for group_index, parallel_group in enumerate(workflow.get('parallel_groups', [])):
            logger.info(f"Executing parallel group {group_index + 1}: {parallel_group}")
            
            # Create tasks for current group
            group_tasks = []
            for task_name in parallel_group:
                task_config = self._get_task_config(task_name, workflow)
                caching_strategy = workflow.get('caching_strategy', {}).get(task_name, 'standard_cache')
                
                task_coroutine = self._execute_single_task(task_name, task_config, caching_strategy, profile)
                group_tasks.append((task_name, task_coroutine))
                total_tasks += 1
            
            # Execute group tasks in parallel
            group_results = await asyncio.gather(
                *[task_coro for _, task_coro in group_tasks],
                return_exceptions=True
            )
            
            # Process results
            for (task_name, _), result in zip(group_tasks, group_results):
                if isinstance(result, Exception):
                    logger.error(f"Task {task_name} failed: {result}")
                    results[task_name] = {'status': 'failed', 'error': str(result)}
                else:
                    results[task_name] = result
                    completed_tasks += 1
                
                # Check for critical failures
                if isinstance(result, Exception) and task_name in ['build', 'security_scan']:
                    logger.critical(f"Critical task {task_name} failed, stopping workflow")
                    return results
        
        logger.info(f"Workflow completed: {completed_tasks}/{total_tasks} tasks successful")
        return results
    
    def _get_task_config(self, task_name: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration for a specific task"""
        for task in workflow.get('tasks', []):
            if task.get('name') == task_name:
                return task
        
        return {'name': task_name, 'command': f'echo "Task {task_name} not configured"'}
    
    async def _execute_single_task(self, task_name: str, task_config: Dict[str, Any], 
                                 caching_strategy: str, profile: OptimizationProfile) -> Dict[str, Any]:
        """Execute a single optimized task"""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_task_cache_key(task_name, task_config)
        
        # Check cache if enabled
        if caching_strategy != 'no_cache' and profile.cache_enabled:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"🚀 Cache hit for task: {task_name}")
                return cached_result
        
        # Execute task
        try:
            command = task_config.get('command', f'echo "Executing {task_name}"')
            timeout = task_config.get('timeout', 300)
            
            # Apply resource limits
            resource_allocation = task_config.get('resource_allocation', {})
            if resource_allocation:
                # In a real implementation, this would apply cgroups or similar
                logger.debug(f"Resource allocation for {task_name}: {resource_allocation}")
            
            # Execute with monitoring
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            result = {
                'status': 'success' if process.returncode == 0 else 'failed',
                'exit_code': process.returncode,
                'execution_time': execution_time,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore'),
                'cache_key': cache_key,
                'resource_usage': await self._collect_task_resource_usage()
            }
            
            # Cache result if successful and appropriate strategy
            if (result['status'] == 'success' and 
                caching_strategy in ['standard_cache', 'aggressive_cache'] and 
                profile.cache_enabled):
                
                cache_ttl = 24 if caching_strategy == 'standard_cache' else 168  # 1 day vs 1 week
                self.cache.put(cache_key, result, ttl_hours=cache_ttl)
            
            logger.info(f"✅ Task completed: {task_name} ({execution_time:.1f}s)")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Task timed out: {task_name}")
            return {
                'status': 'timeout',
                'error': f'Task timed out after {timeout}s',
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            logger.error(f"❌ Task failed: {task_name} - {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _generate_task_cache_key(self, task_name: str, task_config: Dict[str, Any]) -> str:
        """Generate cache key for task"""
        # Include task name, command, and relevant file checksums
        key_components = [
            task_name,
            task_config.get('command', ''),
            str(task_config.get('dependencies', [])),
        ]
        
        # Add file checksums for cache invalidation
        relevant_files = task_config.get('cache_files', [])
        for file_pattern in relevant_files:
            try:
                files = list(self.project_root.glob(file_pattern))
                for file_path in files:
                    if file_path.is_file():
                        mtime = file_path.stat().st_mtime
                        key_components.append(f"{file_path}:{mtime}")
            except Exception:
                pass  # Skip if pattern doesn't match or file doesn't exist
        
        key_string = '|'.join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def _collect_task_resource_usage(self) -> Dict[str, Any]:
        """Collect resource usage during task execution"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_performance_metrics(self, results: Dict[str, Any], execution_time: float, 
                                     profile: OptimizationProfile) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        successful_tasks = sum(1 for r in results.values() if r.get('status') == 'success')
        total_tasks = len(results)
        throughput = total_tasks / execution_time if execution_time > 0 else 0
        
        # Resource efficiency
        total_cpu_time = sum(r.get('execution_time', 0) for r in results.values())
        parallelization_factor = total_cpu_time / execution_time if execution_time > 0 else 1
        resource_efficiency = min(100, (parallelization_factor / profile.parallel_jobs) * 100)
        
        # Cache performance
        cache_stats = self.cache.get_cache_stats()
        cache_hit_rate = cache_stats.get('hit_rate', 0)
        
        # Bottleneck analysis
        task_times = [(name, r.get('execution_time', 0)) for name, r in results.items()]
        task_times.sort(key=lambda x: x[1], reverse=True)
        
        bottlenecks = {
            'slowest_tasks': task_times[:3],
            'total_parallel_efficiency': parallelization_factor,
            'memory_utilization': cache_stats.get('utilization_percent', 0)
        }
        
        return PerformanceMetrics(
            execution_time=execution_time,
            throughput=throughput,
            resource_efficiency=resource_efficiency,
            cache_hit_rate=cache_hit_rate,
            parallelization_factor=parallelization_factor,
            bottleneck_analysis=bottlenecks
        )
    
    def _calculate_optimization_savings(self, execution_time: float, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Calculate optimization savings compared to baseline"""
        # Estimate baseline (sequential execution without optimizations)
        estimated_baseline = execution_time * metrics.parallelization_factor
        
        time_saved = max(0, estimated_baseline - execution_time)
        efficiency_gain = metrics.resource_efficiency
        
        return {
            'estimated_baseline_time': estimated_baseline,
            'actual_execution_time': execution_time,
            'time_saved_seconds': time_saved,
            'time_saved_percent': (time_saved / estimated_baseline) * 100 if estimated_baseline > 0 else 0,
            'efficiency_gain_percent': efficiency_gain,
            'cache_benefit': metrics.cache_hit_rate
        }
    
    def _store_performance_data(self, workflow: Dict[str, Any], results: Dict[str, Any], 
                              metrics: PerformanceMetrics):
        """Store performance data for machine learning and optimization"""
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'workflow_hash': hashlib.sha256(str(workflow).encode()).hexdigest()[:16],
            'metrics': metrics.__dict__,
            'results_summary': {
                'total_tasks': len(results),
                'successful_tasks': sum(1 for r in results.values() if r.get('status') == 'success'),
                'cache_hits': sum(1 for r in results.values() if 'cache_key' in r)
            }
        }
        
        self.performance_history.append(performance_data)
        
        # Save to disk periodically
        if len(self.performance_history) % 10 == 0:
            self._save_performance_history()
    
    def _save_performance_history(self):
        """Save performance history to disk"""
        history_file = self.project_root / 'sdlc_performance_history.json'
        
        try:
            with open(history_file, 'w') as f:
                json.dump(list(self.performance_history), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save performance history: {e}")
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        cache_stats = self.cache.get_cache_stats()
        
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_runs = list(self.performance_history)[-10:]  # Last 10 runs
        
        # Calculate averages
        avg_execution_time = statistics.mean(run['metrics']['execution_time'] for run in recent_runs)
        avg_throughput = statistics.mean(run['metrics']['throughput'] for run in recent_runs)
        avg_efficiency = statistics.mean(run['metrics']['resource_efficiency'] for run in recent_runs)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'optimization_summary': {
                'total_runs_analyzed': len(recent_runs),
                'average_execution_time': avg_execution_time,
                'average_throughput': avg_throughput,
                'average_resource_efficiency': avg_efficiency,
                'optimization_enabled': self.optimization_enabled
            },
            'cache_performance': cache_stats,
            'system_info': self.resource_manager.system_info,
            'recommendations': self._generate_optimization_recommendations(recent_runs)
        }
    
    def _generate_optimization_recommendations(self, recent_runs: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on performance data"""
        recommendations = []
        
        if not recent_runs:
            return ["Insufficient data for recommendations"]
        
        # Analyze patterns
        avg_cache_hit_rate = statistics.mean(run['metrics']['cache_hit_rate'] for run in recent_runs)
        avg_parallelization = statistics.mean(run['metrics']['parallelization_factor'] for run in recent_runs)
        avg_efficiency = statistics.mean(run['metrics']['resource_efficiency'] for run in recent_runs)
        
        if avg_cache_hit_rate < 70:
            recommendations.append("Consider increasing cache size or improving cache key strategies")
        
        if avg_parallelization < 2.0:
            recommendations.append("Look for opportunities to parallelize more tasks")
        
        if avg_efficiency < 60:
            recommendations.append("Consider using a more aggressive optimization profile")
        
        # System-specific recommendations
        system_memory = self.resource_manager.system_info['memory_total_gb']
        if system_memory > 16 and avg_cache_hit_rate < 80:
            recommendations.append("Increase cache size to utilize available memory")
        
        cpu_cores = self.resource_manager.system_info['cpu_cores']
        if cpu_cores > 8 and avg_parallelization < cpu_cores * 0.5:
            recommendations.append("Increase parallel job count to utilize available CPU cores")
        
        if not recommendations:
            recommendations.append("System is well optimized - consider monitoring for regressions")
        
        return recommendations


async def main():
    """Demonstration of hyperscale optimization"""
    print("🚀 HYPERSCALE SDLC OPTIMIZER - Generation 3")
    print("=" * 60)
    
    optimizer = HyperscaleSDLCOptimizer()
    
    # Example workflow definition
    sample_workflow = {
        'name': 'CI/CD Pipeline',
        'complexity': 'high',
        'tasks': [
            {
                'name': 'code_formatting',
                'command': 'python -m black --check src/ tests/',
                'type': 'lint',
                'estimated_time_seconds': 10,
                'resource_intensity': 'low',
                'deterministic': True,
                'dependencies': [],
                'cache_files': ['src/**/*.py', 'tests/**/*.py']
            },
            {
                'name': 'linting',
                'command': 'python -m ruff check src/ tests/',
                'type': 'lint', 
                'estimated_time_seconds': 15,
                'resource_intensity': 'low',
                'deterministic': True,
                'dependencies': [],
                'cache_files': ['src/**/*.py', 'tests/**/*.py']
            },
            {
                'name': 'type_checking',
                'command': 'python -m mypy src/',
                'type': 'lint',
                'estimated_time_seconds': 20,
                'resource_intensity': 'medium',
                'deterministic': True,
                'dependencies': [],
                'cache_files': ['src/**/*.py']
            },
            {
                'name': 'unit_tests',
                'command': 'python -m pytest tests/unit/ -v',
                'type': 'test',
                'estimated_time_seconds': 45,
                'resource_intensity': 'medium',
                'deterministic': True,
                'dependencies': ['code_formatting', 'linting'],
                'cache_files': ['src/**/*.py', 'tests/unit/**/*.py']
            },
            {
                'name': 'integration_tests',
                'command': 'python -m pytest tests/integration/ -v',
                'type': 'test',
                'estimated_time_seconds': 90,
                'resource_intensity': 'high',
                'deterministic': True,
                'dependencies': ['unit_tests'],
                'cache_files': ['src/**/*.py', 'tests/integration/**/*.py']
            },
            {
                'name': 'security_scan',
                'command': 'python -m bandit -r src/ -f json',
                'type': 'security_scan',
                'estimated_time_seconds': 30,
                'resource_intensity': 'medium',
                'deterministic': True,
                'dependencies': [],
                'cache_files': ['src/**/*.py']
            }
        ]
    }
    
    try:
        # Execute optimized workflow
        results = await optimizer.optimize_workflow(sample_workflow)
        
        print(f"\n🎯 OPTIMIZATION RESULTS")
        print(f"Status: {results['status']}")
        print(f"Execution Time: {results['execution_time']:.1f}s")
        print(f"Profile Used: {results['profile_used']}")
        
        metrics = results['metrics']
        print(f"\n📊 PERFORMANCE METRICS")
        print(f"Throughput: {metrics.throughput:.2f} tasks/second")
        print(f"Resource Efficiency: {metrics.resource_efficiency:.1f}%")
        print(f"Cache Hit Rate: {metrics.cache_hit_rate:.1f}%")
        print(f"Parallelization Factor: {metrics.parallelization_factor:.1f}x")
        
        savings = results['optimization_savings']
        print(f"\n💰 OPTIMIZATION SAVINGS")
        print(f"Time Saved: {savings['time_saved_seconds']:.1f}s ({savings['time_saved_percent']:.1f}%)")
        print(f"Efficiency Gain: {savings['efficiency_gain_percent']:.1f}%")
        
        # Generate comprehensive report
        report = optimizer.generate_optimization_report()
        
        print(f"\n🔍 OPTIMIZATION RECOMMENDATIONS")
        for rec in report.get('recommendations', []):
            print(f"• {rec}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())