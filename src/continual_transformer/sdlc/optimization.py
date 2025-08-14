"""Performance optimization and intelligent scaling for SDLC processes."""

import asyncio
import json
import logging
import multiprocessing
import os
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import statistics
import queue
import pickle

from .core import WorkflowTask, WorkflowResult, WorkflowStatus, TaskPriority

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies for workflow execution."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    INTELLIGENT = "intelligent"
    RESOURCE_AWARE = "resource_aware"


class ScalingMode(Enum):
    """Scaling modes for resource allocation."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    PREDICTIVE = "predictive"
    AUTO = "auto"


@dataclass
class ResourceProfile:
    """Resource usage profile for tasks."""
    task_id: str
    avg_cpu_percent: float
    avg_memory_mb: float
    avg_duration_seconds: float
    io_operations: int
    network_usage: float
    preferred_concurrency: int
    resource_intensity: str  # "cpu_bound", "io_bound", "memory_bound", "balanced"
    
    @property
    def efficiency_score(self) -> float:
        """Calculate resource efficiency score (0-100)."""
        # Higher score for faster, less resource-intensive tasks
        duration_score = max(0, 100 - (self.avg_duration_seconds / 10))  # Penalize slow tasks
        resource_score = max(0, 100 - (self.avg_cpu_percent + self.avg_memory_mb / 10))
        return (duration_score + resource_score) / 2


@dataclass
class OptimizationResult:
    """Result of optimization analysis."""
    strategy_used: OptimizationStrategy
    execution_plan: Dict[str, Any]
    estimated_duration: float
    estimated_resource_usage: Dict[str, float]
    parallelization_factor: float
    optimization_confidence: float


class ResourceMonitor:
    """Real-time system resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = []
        self.current_metrics = {}
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.ResourceMonitor")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self.lock:
                    self.current_metrics = metrics
                    self.resource_history.append(metrics)
                    
                    # Keep only recent history (last 1000 entries)
                    if len(self.resource_history) > 1000:
                        self.resource_history.pop(0)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        # Process info
        process = psutil.Process()
        process_info = process.as_dict(attrs=['cpu_percent', 'memory_info', 'num_threads'])
        
        return {
            'timestamp': time.time(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024),
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'process': {
                'cpu_percent': process_info.get('cpu_percent', 0),
                'memory_mb': process_info.get('memory_info', {}).get('rss', 0) / (1024 * 1024),
                'threads': process_info.get('num_threads', 1)
            }
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        with self.lock:
            return self.current_metrics.copy()
    
    def get_resource_trends(self, minutes: int = 10) -> Dict[str, List[float]]:
        """Get resource trends over time."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            recent_metrics = [
                m for m in self.resource_history 
                if m['timestamp'] >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        return {
            'cpu_trend': [m['system']['cpu_percent'] for m in recent_metrics],
            'memory_trend': [m['system']['memory_percent'] for m in recent_metrics],
            'timestamps': [m['timestamp'] for m in recent_metrics]
        }
    
    def get_resource_availability(self) -> Dict[str, float]:
        """Get current resource availability (0-1 scale)."""
        metrics = self.get_current_metrics()
        if not metrics:
            return {'cpu': 0.5, 'memory': 0.5, 'disk': 0.5}  # Conservative defaults
        
        system = metrics.get('system', {})
        
        return {
            'cpu': max(0, (100 - system.get('cpu_percent', 50)) / 100),
            'memory': max(0, (100 - system.get('memory_percent', 50)) / 100),
            'disk': max(0, (100 - system.get('disk_percent', 50)) / 100)
        }


class TaskProfiler:
    """Profiler for analyzing task resource usage patterns."""
    
    def __init__(self):
        self.task_profiles = {}
        self.execution_history = []
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.TaskProfiler")
    
    def record_execution(self, task: WorkflowTask, result: WorkflowResult, resource_metrics: Dict[str, Any]):
        """Record task execution for profiling."""
        with self.lock:
            # Update task profile
            if task.id not in self.task_profiles:
                self.task_profiles[task.id] = {
                    'executions': [],
                    'avg_metrics': {},
                    'resource_intensity': 'balanced',
                    'preferred_concurrency': 1
                }
            
            profile = self.task_profiles[task.id]
            
            # Record this execution
            execution_record = {
                'timestamp': result.end_time,
                'duration': result.duration,
                'success': result.status == WorkflowStatus.COMPLETED,
                'cpu_percent': resource_metrics.get('cpu_percent', 0),
                'memory_mb': resource_metrics.get('memory_mb', 0),
                'resource_metrics': resource_metrics
            }
            
            profile['executions'].append(execution_record)
            
            # Keep only recent executions (last 100)
            if len(profile['executions']) > 100:
                profile['executions'].pop(0)
            
            # Update aggregated metrics
            self._update_profile_metrics(task.id)
            
            # Add to global history
            self.execution_history.append({
                'task_id': task.id,
                'task_name': task.name,
                'execution_record': execution_record
            })
            
            # Keep global history manageable
            if len(self.execution_history) > 1000:
                self.execution_history.pop(0)
    
    def _update_profile_metrics(self, task_id: str):
        """Update aggregated metrics for a task profile."""
        profile = self.task_profiles[task_id]
        executions = profile['executions']
        
        if not executions:
            return
        
        # Calculate averages
        successful_executions = [e for e in executions if e['success']]
        if not successful_executions:
            return
        
        profile['avg_metrics'] = {
            'duration': statistics.mean(e['duration'] for e in successful_executions),
            'cpu_percent': statistics.mean(e['cpu_percent'] for e in successful_executions),
            'memory_mb': statistics.mean(e['memory_mb'] for e in successful_executions),
            'success_rate': len(successful_executions) / len(executions) * 100
        }
        
        # Determine resource intensity
        avg_cpu = profile['avg_metrics']['cpu_percent']
        avg_memory = profile['avg_metrics']['memory_mb']
        
        if avg_cpu > 70:
            profile['resource_intensity'] = 'cpu_bound'
            profile['preferred_concurrency'] = max(1, multiprocessing.cpu_count() // 2)
        elif avg_memory > 1000:  # > 1GB
            profile['resource_intensity'] = 'memory_bound'
            profile['preferred_concurrency'] = max(1, multiprocessing.cpu_count() // 4)
        elif profile['avg_metrics']['duration'] > 60:  # Long running tasks
            profile['resource_intensity'] = 'io_bound'
            profile['preferred_concurrency'] = multiprocessing.cpu_count() * 2
        else:
            profile['resource_intensity'] = 'balanced'
            profile['preferred_concurrency'] = multiprocessing.cpu_count()
    
    def get_task_profile(self, task_id: str) -> Optional[ResourceProfile]:
        """Get resource profile for a task."""
        with self.lock:
            if task_id not in self.task_profiles:
                return None
            
            profile = self.task_profiles[task_id]
            avg_metrics = profile.get('avg_metrics', {})
            
            if not avg_metrics:
                return None
            
            return ResourceProfile(
                task_id=task_id,
                avg_cpu_percent=avg_metrics.get('cpu_percent', 0),
                avg_memory_mb=avg_metrics.get('memory_mb', 0),
                avg_duration_seconds=avg_metrics.get('duration', 0),
                io_operations=0,  # Would need more detailed monitoring
                network_usage=0,  # Would need more detailed monitoring
                preferred_concurrency=profile.get('preferred_concurrency', 1),
                resource_intensity=profile.get('resource_intensity', 'balanced')
            )
    
    def get_all_profiles(self) -> Dict[str, ResourceProfile]:
        """Get all task profiles."""
        profiles = {}
        with self.lock:
            for task_id in self.task_profiles.keys():
                profile = self.get_task_profile(task_id)
                if profile:
                    profiles[task_id] = profile
        return profiles


class IntelligentScheduler:
    """Intelligent task scheduler with optimization."""
    
    def __init__(self, resource_monitor: ResourceMonitor, task_profiler: TaskProfiler):
        self.resource_monitor = resource_monitor
        self.task_profiler = task_profiler
        self.optimization_history = []
        
        self.logger = logging.getLogger(f"{__name__}.IntelligentScheduler")
    
    def optimize_execution_plan(
        self, 
        tasks: List[WorkflowTask], 
        strategy: OptimizationStrategy = OptimizationStrategy.INTELLIGENT
    ) -> OptimizationResult:
        """Create optimized execution plan for tasks."""
        
        self.logger.info(f"Optimizing execution plan for {len(tasks)} tasks using {strategy.value} strategy")
        
        if strategy == OptimizationStrategy.SEQUENTIAL:
            return self._sequential_plan(tasks)
        elif strategy == OptimizationStrategy.PARALLEL:
            return self._parallel_plan(tasks)
        elif strategy == OptimizationStrategy.ADAPTIVE:
            return self._adaptive_plan(tasks)
        elif strategy == OptimizationStrategy.INTELLIGENT:
            return self._intelligent_plan(tasks)
        elif strategy == OptimizationStrategy.RESOURCE_AWARE:
            return self._resource_aware_plan(tasks)
        else:
            return self._sequential_plan(tasks)
    
    def _sequential_plan(self, tasks: List[WorkflowTask]) -> OptimizationResult:
        """Create sequential execution plan."""
        sorted_tasks = sorted(tasks, key=lambda t: (len(t.dependencies), t.priority.value), reverse=True)
        
        estimated_duration = 0
        for task in sorted_tasks:
            profile = self.task_profiler.get_task_profile(task.id)
            if profile:
                estimated_duration += profile.avg_duration_seconds
            else:
                estimated_duration += 60  # Default estimate
        
        return OptimizationResult(
            strategy_used=OptimizationStrategy.SEQUENTIAL,
            execution_plan={
                'execution_order': [t.id for t in sorted_tasks],
                'parallelization': {},
                'resource_allocation': {'max_workers': 1}
            },
            estimated_duration=estimated_duration,
            estimated_resource_usage={'cpu': 25, 'memory': 30},
            parallelization_factor=1.0,
            optimization_confidence=0.9
        )
    
    def _parallel_plan(self, tasks: List[WorkflowTask]) -> OptimizationResult:
        """Create parallel execution plan."""
        # Group tasks by dependency levels
        dependency_levels = self._analyze_dependencies(tasks)
        
        max_parallel = min(len(tasks), multiprocessing.cpu_count())
        
        estimated_duration = 0
        for level_tasks in dependency_levels:
            level_duration = 0
            if level_tasks:
                # Estimate duration for this level (tasks run in parallel)
                durations = []
                for task in level_tasks:
                    profile = self.task_profiler.get_task_profile(task.id)
                    duration = profile.avg_duration_seconds if profile else 60
                    durations.append(duration)
                level_duration = max(durations) if durations else 0
            estimated_duration += level_duration
        
        return OptimizationResult(
            strategy_used=OptimizationStrategy.PARALLEL,
            execution_plan={
                'dependency_levels': [[t.id for t in level] for level in dependency_levels],
                'parallelization': {'max_workers': max_parallel},
                'resource_allocation': {'max_workers': max_parallel}
            },
            estimated_duration=estimated_duration,
            estimated_resource_usage={'cpu': 70, 'memory': 60},
            parallelization_factor=max_parallel,
            optimization_confidence=0.75
        )
    
    def _adaptive_plan(self, tasks: List[WorkflowTask]) -> OptimizationResult:
        """Create adaptive execution plan based on current system state."""
        resource_availability = self.resource_monitor.get_resource_availability()
        
        # Adjust parallelization based on available resources
        cpu_available = resource_availability.get('cpu', 0.5)
        memory_available = resource_availability.get('memory', 0.5)
        
        max_workers = max(1, int(multiprocessing.cpu_count() * min(cpu_available, memory_available)))
        
        # Use parallel strategy but with dynamic worker count
        base_plan = self._parallel_plan(tasks)
        
        # Adjust resource allocation
        base_plan.execution_plan['resource_allocation']['max_workers'] = max_workers
        base_plan.parallelization_factor = max_workers
        base_plan.strategy_used = OptimizationStrategy.ADAPTIVE
        base_plan.optimization_confidence = 0.8
        
        # Adjust estimated duration based on available resources
        resource_factor = min(cpu_available, memory_available)
        base_plan.estimated_duration = base_plan.estimated_duration / max(0.1, resource_factor)
        
        return base_plan
    
    def _intelligent_plan(self, tasks: List[WorkflowTask]) -> OptimizationResult:
        """Create intelligent execution plan using ML-like optimization."""
        # Analyze task profiles and dependencies
        dependency_levels = self._analyze_dependencies(tasks)
        resource_availability = self.resource_monitor.get_resource_availability()
        
        # Get task profiles
        task_profiles = {}
        for task in tasks:
            profile = self.task_profiler.get_task_profile(task.id)
            task_profiles[task.id] = profile
        
        # Optimize each dependency level
        optimized_levels = []
        total_duration = 0
        
        for level_tasks in dependency_levels:
            level_plan = self._optimize_task_level(level_tasks, task_profiles, resource_availability)
            optimized_levels.append(level_plan)
            total_duration += level_plan['estimated_duration']
        
        # Calculate parallelization factor
        total_tasks = len(tasks)
        avg_parallel = statistics.mean(level['max_workers'] for level in optimized_levels)
        
        return OptimizationResult(
            strategy_used=OptimizationStrategy.INTELLIGENT,
            execution_plan={
                'optimized_levels': optimized_levels,
                'resource_allocation': {'adaptive': True},
                'intelligent_routing': True
            },
            estimated_duration=total_duration,
            estimated_resource_usage={'cpu': 60, 'memory': 50},
            parallelization_factor=avg_parallel,
            optimization_confidence=0.85
        )
    
    def _resource_aware_plan(self, tasks: List[WorkflowTask]) -> OptimizationResult:
        """Create resource-aware execution plan."""
        # Group tasks by resource intensity
        cpu_bound_tasks = []
        io_bound_tasks = []
        memory_bound_tasks = []
        balanced_tasks = []
        
        for task in tasks:
            profile = self.task_profiler.get_task_profile(task.id)
            if profile:
                if profile.resource_intensity == 'cpu_bound':
                    cpu_bound_tasks.append(task)
                elif profile.resource_intensity == 'io_bound':
                    io_bound_tasks.append(task)
                elif profile.resource_intensity == 'memory_bound':
                    memory_bound_tasks.append(task)
                else:
                    balanced_tasks.append(task)
            else:
                balanced_tasks.append(task)
        
        # Create execution plan that optimizes resource usage
        execution_plan = {
            'resource_groups': {
                'cpu_bound': [t.id for t in cpu_bound_tasks],
                'io_bound': [t.id for t in io_bound_tasks],
                'memory_bound': [t.id for t in memory_bound_tasks],
                'balanced': [t.id for t in balanced_tasks]
            },
            'execution_strategy': 'resource_optimized',
            'worker_allocation': {
                'cpu_bound': max(1, multiprocessing.cpu_count() // 2),
                'io_bound': multiprocessing.cpu_count() * 2,
                'memory_bound': max(1, multiprocessing.cpu_count() // 4),
                'balanced': multiprocessing.cpu_count()
            }
        }
        
        # Estimate duration based on resource grouping
        estimated_duration = self._estimate_resource_aware_duration(
            cpu_bound_tasks, io_bound_tasks, memory_bound_tasks, balanced_tasks
        )
        
        return OptimizationResult(
            strategy_used=OptimizationStrategy.RESOURCE_AWARE,
            execution_plan=execution_plan,
            estimated_duration=estimated_duration,
            estimated_resource_usage={'cpu': 80, 'memory': 70},
            parallelization_factor=multiprocessing.cpu_count(),
            optimization_confidence=0.8
        )
    
    def _analyze_dependencies(self, tasks: List[WorkflowTask]) -> List[List[WorkflowTask]]:
        """Analyze task dependencies and group into executable levels."""
        task_map = {task.id: task for task in tasks}
        levels = []
        remaining_tasks = set(task.id for task in tasks)
        completed_tasks = set()
        
        while remaining_tasks:
            current_level = []
            
            # Find tasks that can be executed (dependencies satisfied)
            for task_id in list(remaining_tasks):
                task = task_map[task_id]
                if all(dep in completed_tasks for dep in task.dependencies):
                    current_level.append(task)
                    remaining_tasks.remove(task_id)
            
            if not current_level:
                # Circular dependency or other issue
                self.logger.warning(f"Possible circular dependency detected with remaining tasks: {remaining_tasks}")
                # Add remaining tasks as final level
                current_level = [task_map[task_id] for task_id in remaining_tasks]
                remaining_tasks.clear()
            
            levels.append(current_level)
            completed_tasks.update(task.id for task in current_level)
        
        return levels
    
    def _optimize_task_level(
        self, 
        tasks: List[WorkflowTask], 
        profiles: Dict[str, ResourceProfile], 
        resource_availability: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize execution for a single dependency level."""
        
        # Analyze task characteristics
        cpu_intensive_count = sum(1 for t in tasks if profiles.get(t.id) and profiles[t.id].resource_intensity == 'cpu_bound')
        io_intensive_count = sum(1 for t in tasks if profiles.get(t.id) and profiles[t.id].resource_intensity == 'io_bound')
        
        # Determine optimal worker count
        if io_intensive_count > cpu_intensive_count:
            # More I/O bound tasks, can use more workers
            max_workers = min(len(tasks), multiprocessing.cpu_count() * 2)
        else:
            # More CPU bound tasks, limit workers
            max_workers = min(len(tasks), multiprocessing.cpu_count())
        
        # Adjust for resource availability
        cpu_available = resource_availability.get('cpu', 0.5)
        memory_available = resource_availability.get('memory', 0.5)
        
        max_workers = max(1, int(max_workers * min(cpu_available, memory_available)))
        
        # Estimate duration
        if profiles:
            durations = [profiles[t.id].avg_duration_seconds for t in tasks if profiles.get(t.id)]
            estimated_duration = max(durations) if durations else 60
        else:
            estimated_duration = 60
        
        return {
            'tasks': [t.id for t in tasks],
            'max_workers': max_workers,
            'estimated_duration': estimated_duration,
            'optimization_notes': f'Level with {len(tasks)} tasks, {cpu_intensive_count} CPU-bound, {io_intensive_count} I/O-bound'
        }
    
    def _estimate_resource_aware_duration(
        self, 
        cpu_bound: List[WorkflowTask],
        io_bound: List[WorkflowTask],
        memory_bound: List[WorkflowTask],
        balanced: List[WorkflowTask]
    ) -> float:
        """Estimate duration for resource-aware execution."""
        
        # These can run in parallel with different worker pools
        group_durations = []
        
        for task_group, group_name in [
            (cpu_bound, 'cpu_bound'),
            (io_bound, 'io_bound'),
            (memory_bound, 'memory_bound'),
            (balanced, 'balanced')
        ]:
            if task_group:
                total_work = 0
                for task in task_group:
                    profile = self.task_profiler.get_task_profile(task.id)
                    duration = profile.avg_duration_seconds if profile else 60
                    total_work += duration
                
                # Estimate parallel execution time
                if group_name == 'io_bound':
                    workers = multiprocessing.cpu_count() * 2
                elif group_name == 'cpu_bound':
                    workers = max(1, multiprocessing.cpu_count() // 2)
                elif group_name == 'memory_bound':
                    workers = max(1, multiprocessing.cpu_count() // 4)
                else:
                    workers = multiprocessing.cpu_count()
                
                group_duration = total_work / workers
                group_durations.append(group_duration)
        
        # Groups run in parallel, so take the maximum
        return max(group_durations) if group_durations else 0


class OptimizedWorkflowEngine:
    """Enhanced workflow engine with performance optimization."""
    
    def __init__(self, max_workers: int = None, optimization_strategy: OptimizationStrategy = OptimizationStrategy.INTELLIGENT):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.optimization_strategy = optimization_strategy
        
        # Initialize components
        self.resource_monitor = ResourceMonitor()
        self.task_profiler = TaskProfiler()
        self.intelligent_scheduler = IntelligentScheduler(self.resource_monitor, self.task_profiler)
        
        # Execution state
        self.active_workflows = {}
        self.completed_workflows = {}
        
        self.logger = logging.getLogger(f"{__name__}.OptimizedWorkflowEngine")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
    
    def execute_optimized_workflow(self, tasks: List[WorkflowTask]) -> Dict[str, WorkflowResult]:
        """Execute workflow with optimization."""
        workflow_id = f"optimized_workflow_{int(time.time())}"
        
        self.logger.info(f"Starting optimized workflow execution: {workflow_id}")
        
        # Create optimization plan
        optimization_result = self.intelligent_scheduler.optimize_execution_plan(
            tasks, self.optimization_strategy
        )
        
        self.logger.info(
            f"Optimization plan created: strategy={optimization_result.strategy_used.value}, "
            f"estimated_duration={optimization_result.estimated_duration:.2f}s, "
            f"parallelization_factor={optimization_result.parallelization_factor:.2f}"
        )
        
        # Execute according to plan
        start_time = time.time()
        
        try:
            if optimization_result.strategy_used == OptimizationStrategy.SEQUENTIAL:
                results = self._execute_sequential(tasks, optimization_result)
            elif optimization_result.strategy_used == OptimizationStrategy.PARALLEL:
                results = self._execute_parallel(tasks, optimization_result)
            elif optimization_result.strategy_used == OptimizationStrategy.ADAPTIVE:
                results = self._execute_adaptive(tasks, optimization_result)
            elif optimization_result.strategy_used == OptimizationStrategy.INTELLIGENT:
                results = self._execute_intelligent(tasks, optimization_result)
            elif optimization_result.strategy_used == OptimizationStrategy.RESOURCE_AWARE:
                results = self._execute_resource_aware(tasks, optimization_result)
            else:
                results = self._execute_sequential(tasks, optimization_result)
            
            execution_duration = time.time() - start_time
            
            self.logger.info(
                f"Workflow {workflow_id} completed in {execution_duration:.2f}s "
                f"(estimated: {optimization_result.estimated_duration:.2f}s)"
            )
            
            # Record performance metrics
            self._record_workflow_performance(
                workflow_id, tasks, results, optimization_result, execution_duration
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimized workflow execution failed: {e}")
            raise
    
    def _execute_sequential(self, tasks: List[WorkflowTask], optimization_result: OptimizationResult) -> Dict[str, WorkflowResult]:
        """Execute tasks sequentially."""
        results = {}
        execution_order = optimization_result.execution_plan.get('execution_order', [t.id for t in tasks])
        task_map = {t.id: t for t in tasks}
        
        for task_id in execution_order:
            if task_id in task_map:
                task = task_map[task_id]
                result = self._execute_single_task_with_monitoring(task)
                results[task_id] = result
        
        return results
    
    def _execute_parallel(self, tasks: List[WorkflowTask], optimization_result: OptimizationResult) -> Dict[str, WorkflowResult]:
        """Execute tasks in parallel by dependency levels."""
        results = {}
        dependency_levels = optimization_result.execution_plan.get('dependency_levels', [[t.id for t in tasks]])
        task_map = {t.id: t for t in tasks}
        max_workers = optimization_result.execution_plan.get('resource_allocation', {}).get('max_workers', self.max_workers)
        
        for level in dependency_levels:
            level_tasks = [task_map[task_id] for task_id in level if task_id in task_map]
            level_results = self._execute_task_level_parallel(level_tasks, max_workers)
            results.update(level_results)
        
        return results
    
    def _execute_adaptive(self, tasks: List[WorkflowTask], optimization_result: OptimizationResult) -> Dict[str, WorkflowResult]:
        """Execute tasks with adaptive resource management."""
        # Similar to parallel but with dynamic worker adjustment
        results = {}
        dependency_levels = optimization_result.execution_plan.get('dependency_levels', [[t.id for t in tasks]])
        task_map = {t.id: t for t in tasks}
        
        for level in dependency_levels:
            # Adapt worker count based on current resource availability
            resource_availability = self.resource_monitor.get_resource_availability()
            adaptive_workers = max(1, int(self.max_workers * min(resource_availability.values())))
            
            level_tasks = [task_map[task_id] for task_id in level if task_id in task_map]
            level_results = self._execute_task_level_parallel(level_tasks, adaptive_workers)
            results.update(level_results)
        
        return results
    
    def _execute_intelligent(self, tasks: List[WorkflowTask], optimization_result: OptimizationResult) -> Dict[str, WorkflowResult]:
        """Execute tasks with intelligent optimization."""
        results = {}
        optimized_levels = optimization_result.execution_plan.get('optimized_levels', [])
        task_map = {t.id: t for t in tasks}
        
        for level_plan in optimized_levels:
            level_task_ids = level_plan.get('tasks', [])
            max_workers = level_plan.get('max_workers', self.max_workers)
            
            level_tasks = [task_map[task_id] for task_id in level_task_ids if task_id in task_map]
            
            self.logger.info(f"Executing level with {len(level_tasks)} tasks using {max_workers} workers")
            
            level_results = self._execute_task_level_parallel(level_tasks, max_workers)
            results.update(level_results)
        
        return results
    
    def _execute_resource_aware(self, tasks: List[WorkflowTask], optimization_result: OptimizationResult) -> Dict[str, WorkflowResult]:
        """Execute tasks with resource-aware grouping."""
        results = {}
        resource_groups = optimization_result.execution_plan.get('resource_groups', {})
        worker_allocation = optimization_result.execution_plan.get('worker_allocation', {})
        task_map = {t.id: t for t in tasks}
        
        # Execute each resource group with appropriate worker count
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for group_name, task_ids in resource_groups.items():
                if task_ids:
                    group_tasks = [task_map[task_id] for task_id in task_ids if task_id in task_map]
                    workers = worker_allocation.get(group_name, self.max_workers)
                    
                    future = executor.submit(self._execute_task_level_parallel, group_tasks, workers)
                    futures[group_name] = future
            
            # Collect results
            for group_name, future in futures.items():
                try:
                    group_results = future.result()
                    results.update(group_results)
                except Exception as e:
                    self.logger.error(f"Resource group {group_name} execution failed: {e}")
        
        return results
    
    def _execute_task_level_parallel(self, tasks: List[WorkflowTask], max_workers: int) -> Dict[str, WorkflowResult]:
        """Execute a level of tasks in parallel."""
        results = {}
        
        if not tasks:
            return results
        
        if len(tasks) == 1:
            # Single task, execute directly
            task = tasks[0]
            result = self._execute_single_task_with_monitoring(task)
            results[task.id] = result
            return results
        
        # Multiple tasks, use thread pool
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._execute_single_task_with_monitoring, task): task
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results[task.id] = result
                except Exception as e:
                    self.logger.error(f"Task {task.id} execution failed: {e}")
                    # Create failed result
                    results[task.id] = WorkflowResult(
                        task_id=task.id,
                        status=WorkflowStatus.FAILED,
                        output="",
                        error=str(e),
                        duration=0,
                        start_time=time.time(),
                        end_time=time.time(),
                        exit_code=1
                    )
        
        return results
    
    def _execute_single_task_with_monitoring(self, task: WorkflowTask) -> WorkflowResult:
        """Execute single task with resource monitoring."""
        start_time = time.time()
        start_metrics = self.resource_monitor.get_current_metrics()
        
        try:
            # Execute task (this would be the actual task execution logic)
            result = self._execute_task_basic(task)
            
            end_time = time.time()
            end_metrics = self.resource_monitor.get_current_metrics()
            
            # Calculate resource usage during task execution
            resource_usage = self._calculate_resource_usage(start_metrics, end_metrics, result.duration)
            
            # Record in task profiler
            self.task_profiler.record_execution(task, result, resource_usage)
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            return WorkflowResult(
                task_id=task.id,
                status=WorkflowStatus.FAILED,
                output="",
                error=str(e),
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                exit_code=1
            )
    
    def _execute_task_basic(self, task: WorkflowTask) -> WorkflowResult:
        """Basic task execution (same as in core module)."""
        import subprocess
        
        start_time = time.time()
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(task.environment or {})
            
            # Set working directory
            cwd = task.working_dir or os.getcwd()
            
            # Execute command
            result = subprocess.run(
                task.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=task.timeout,
                env=env,
                cwd=cwd
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            status = WorkflowStatus.COMPLETED if result.returncode == 0 else WorkflowStatus.FAILED
            
            return WorkflowResult(
                task_id=task.id,
                status=status,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                exit_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            
            return WorkflowResult(
                task_id=task.id,
                status=WorkflowStatus.FAILED,
                output="",
                error=f"Task timed out after {task.timeout} seconds",
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                exit_code=124
            )
        
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            return WorkflowResult(
                task_id=task.id,
                status=WorkflowStatus.FAILED,
                output="",
                error=str(e),
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                exit_code=1
            )
    
    def _calculate_resource_usage(
        self, 
        start_metrics: Dict[str, Any], 
        end_metrics: Dict[str, Any], 
        duration: float
    ) -> Dict[str, float]:
        """Calculate resource usage during task execution."""
        
        if not start_metrics or not end_metrics:
            return {'cpu_percent': 0, 'memory_mb': 0}
        
        start_process = start_metrics.get('process', {})
        end_process = end_metrics.get('process', {})
        
        # Calculate average resource usage
        avg_cpu = (start_process.get('cpu_percent', 0) + end_process.get('cpu_percent', 0)) / 2
        avg_memory = (start_process.get('memory_mb', 0) + end_process.get('memory_mb', 0)) / 2
        
        return {
            'cpu_percent': avg_cpu,
            'memory_mb': avg_memory,
            'duration': duration
        }
    
    def _record_workflow_performance(
        self,
        workflow_id: str,
        tasks: List[WorkflowTask],
        results: Dict[str, WorkflowResult],
        optimization_result: OptimizationResult,
        actual_duration: float
    ):
        """Record workflow performance metrics."""
        
        performance_record = {
            'workflow_id': workflow_id,
            'task_count': len(tasks),
            'optimization_strategy': optimization_result.strategy_used.value,
            'estimated_duration': optimization_result.estimated_duration,
            'actual_duration': actual_duration,
            'duration_accuracy': abs(optimization_result.estimated_duration - actual_duration) / max(optimization_result.estimated_duration, 1),
            'success_rate': sum(1 for r in results.values() if r.status == WorkflowStatus.COMPLETED) / len(results) * 100,
            'parallelization_factor': optimization_result.parallelization_factor,
            'timestamp': time.time()
        }
        
        # Store performance record
        self.completed_workflows[workflow_id] = performance_record
        
        self.logger.info(f"Recorded performance for workflow {workflow_id}: {performance_record}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.completed_workflows:
            return {}
        
        workflows = list(self.completed_workflows.values())
        
        return {
            'total_workflows': len(workflows),
            'avg_duration': statistics.mean(w['actual_duration'] for w in workflows),
            'avg_success_rate': statistics.mean(w['success_rate'] for w in workflows),
            'avg_parallelization': statistics.mean(w['parallelization_factor'] for w in workflows),
            'duration_prediction_accuracy': 100 - statistics.mean(w['duration_accuracy'] * 100 for w in workflows),
            'optimization_strategies_used': {
                strategy: sum(1 for w in workflows if w['optimization_strategy'] == strategy)
                for strategy in set(w['optimization_strategy'] for w in workflows)
            },
            'resource_profiles': len(self.task_profiler.get_all_profiles()),
            'system_health': self.resource_monitor.get_current_metrics()
        }
    
    def shutdown(self):
        """Shutdown the optimized workflow engine."""
        self.resource_monitor.stop_monitoring()
        self.logger.info("Optimized workflow engine shut down")