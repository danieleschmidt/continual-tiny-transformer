"""Core SDLC management and workflow engine."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task execution priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkflowTask:
    """Individual task in SDLC workflow."""
    id: str
    name: str
    command: str
    description: str
    dependencies: List[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: int = 300  # seconds
    retry_count: int = 3
    environment: Dict[str, str] = None
    working_dir: str = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.environment is None:
            self.environment = {}


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    task_id: str
    status: WorkflowStatus
    output: str
    error: str
    duration: float
    start_time: float
    end_time: float
    exit_code: int = 0


class WorkflowEngine:
    """Core workflow execution engine."""
    
    def __init__(self, max_workers: int = 4, log_level: str = "INFO"):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_results = {}
        self.lock = threading.Lock()
        
        # Configure logging
        self.logger = logging.getLogger(f"{__name__}.WorkflowEngine")
        self.logger.setLevel(getattr(logging, log_level.upper()))
    
    def execute_task(self, task: WorkflowTask) -> WorkflowResult:
        """Execute a single workflow task."""
        start_time = time.time()
        
        with self.lock:
            self.running_tasks[task.id] = task
        
        self.logger.info(f"Starting task: {task.name} ({task.id})")
        
        try:
            # Prepare environment
            env = {}
            env.update(task.environment)
            
            # Set working directory
            cwd = task.working_dir or str(Path.cwd())
            
            # Execute command with retries
            result = None
            last_error = None
            
            for attempt in range(task.retry_count):
                try:
                    result = subprocess.run(
                        task.command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=task.timeout,
                        env=env,
                        cwd=cwd
                    )
                    
                    # If successful, break retry loop
                    if result.returncode == 0:
                        break
                    else:
                        last_error = f"Command failed with exit code {result.returncode}"
                        if attempt < task.retry_count - 1:
                            self.logger.warning(f"Task {task.id} attempt {attempt + 1} failed, retrying...")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        
                except subprocess.TimeoutExpired as e:
                    last_error = f"Task timed out after {task.timeout} seconds"
                    self.logger.error(f"Task {task.id} timed out on attempt {attempt + 1}")
                except Exception as e:
                    last_error = str(e)
                    self.logger.error(f"Task {task.id} failed on attempt {attempt + 1}: {e}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Create result
            if result and result.returncode == 0:
                status = WorkflowStatus.COMPLETED
                output = result.stdout
                error = result.stderr
                exit_code = result.returncode
                self.logger.info(f"Task {task.id} completed successfully in {duration:.2f}s")
            else:
                status = WorkflowStatus.FAILED
                output = result.stdout if result else ""
                error = last_error or (result.stderr if result else "Unknown error")
                exit_code = result.returncode if result else 1
                self.logger.error(f"Task {task.id} failed after {task.retry_count} attempts")
            
            workflow_result = WorkflowResult(
                task_id=task.id,
                status=status,
                output=output,
                error=error,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                exit_code=exit_code
            )
            
            # Update task tracking
            with self.lock:
                del self.running_tasks[task.id]
                self.task_results[task.id] = workflow_result
                
                if status == WorkflowStatus.COMPLETED:
                    self.completed_tasks[task.id] = task
                else:
                    self.failed_tasks[task.id] = task
            
            return workflow_result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.error(f"Unexpected error executing task {task.id}: {e}")
            
            workflow_result = WorkflowResult(
                task_id=task.id,
                status=WorkflowStatus.FAILED,
                output="",
                error=str(e),
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                exit_code=1
            )
            
            with self.lock:
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                self.task_results[task.id] = workflow_result
                self.failed_tasks[task.id] = task
            
            return workflow_result
    
    def execute_workflow(self, tasks: List[WorkflowTask]) -> Dict[str, WorkflowResult]:
        """Execute workflow with dependency resolution."""
        self.logger.info(f"Starting workflow with {len(tasks)} tasks")
        
        # Build dependency graph
        task_map = {task.id: task for task in tasks}
        dependency_graph = self._build_dependency_graph(tasks)
        
        # Execute tasks in dependency order
        results = {}
        completed = set()
        failed = set()
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value, reverse=True)
        
        while len(completed) + len(failed) < len(tasks):
            # Find ready tasks (dependencies satisfied)
            ready_tasks = []
            
            for task in sorted_tasks:
                if (task.id not in completed and 
                    task.id not in failed and 
                    task.id not in self.running_tasks):
                    
                    # Check if all dependencies are completed
                    if all(dep in completed for dep in task.dependencies):
                        ready_tasks.append(task)
            
            if not ready_tasks:
                # Check if any tasks are still running
                if not self.running_tasks:
                    self.logger.error("No ready tasks and no running tasks - possible circular dependency")
                    break
                
                # Wait for running tasks to complete
                time.sleep(0.5)
                continue
            
            # Submit ready tasks for execution
            futures = {}
            for task in ready_tasks:
                if len(self.running_tasks) < self.max_workers:
                    future = self.executor.submit(self.execute_task, task)
                    futures[task.id] = future
            
            # Wait for at least one task to complete
            if futures:
                completed_futures = []
                while not completed_futures:
                    for task_id, future in futures.items():
                        if future.done():
                            completed_futures.append((task_id, future))
                    
                    if not completed_futures:
                        time.sleep(0.1)
                
                # Process completed futures
                for task_id, future in completed_futures:
                    try:
                        result = future.result()
                        results[task_id] = result
                        
                        if result.status == WorkflowStatus.COMPLETED:
                            completed.add(task_id)
                        else:
                            failed.add(task_id)
                            
                            # Check if this failure should stop the workflow
                            if task_map[task_id].priority == TaskPriority.CRITICAL:
                                self.logger.error(f"Critical task {task_id} failed, stopping workflow")
                                return results
                                
                    except Exception as e:
                        self.logger.error(f"Error getting result for task {task_id}: {e}")
                        failed.add(task_id)
        
        self.logger.info(f"Workflow completed: {len(completed)} successful, {len(failed)} failed")
        return results
    
    def _build_dependency_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build task dependency graph."""
        graph = {}
        for task in tasks:
            graph[task.id] = task.dependencies.copy()
        return graph
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow engine status."""
        with self.lock:
            return {
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "max_workers": self.max_workers,
                "total_results": len(self.task_results)
            }
    
    def shutdown(self):
        """Shutdown the workflow engine."""
        self.logger.info("Shutting down workflow engine")
        self.executor.shutdown(wait=True)


class SDLCManager:
    """Main SDLC management orchestrator."""
    
    def __init__(self, project_path: str, config: Optional[Dict] = None):
        self.project_path = Path(project_path)
        self.config = config or {}
        self.workflow_engine = WorkflowEngine(
            max_workers=self.config.get('max_workers', 4)
        )
        self.workflows = {}
        self.logger = logging.getLogger(f"{__name__}.SDLCManager")
    
    def create_workflow(self, name: str, tasks: List[WorkflowTask]) -> str:
        """Create a new workflow."""
        workflow_id = f"{name}_{int(time.time())}"
        
        self.workflows[workflow_id] = {
            'name': name,
            'tasks': tasks,
            'created_at': time.time(),
            'status': WorkflowStatus.PENDING
        }
        
        self.logger.info(f"Created workflow: {name} ({workflow_id})")
        return workflow_id
    
    def execute_workflow(self, workflow_id: str) -> Dict[str, WorkflowResult]:
        """Execute a workflow by ID."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        workflow['status'] = WorkflowStatus.RUNNING
        workflow['start_time'] = time.time()
        
        self.logger.info(f"Executing workflow: {workflow['name']}")
        
        try:
            results = self.workflow_engine.execute_workflow(workflow['tasks'])
            
            # Determine overall workflow status
            if all(r.status == WorkflowStatus.COMPLETED for r in results.values()):
                workflow['status'] = WorkflowStatus.COMPLETED
            else:
                workflow['status'] = WorkflowStatus.FAILED
            
            workflow['end_time'] = time.time()
            workflow['results'] = results
            
            self.logger.info(f"Workflow {workflow['name']} completed with status: {workflow['status']}")
            return results
            
        except Exception as e:
            workflow['status'] = WorkflowStatus.FAILED
            workflow['error'] = str(e)
            workflow['end_time'] = time.time()
            
            self.logger.error(f"Workflow {workflow['name']} failed: {e}")
            raise
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        return self.workflows[workflow_id]
    
    def create_standard_ci_workflow(self) -> str:
        """Create standard CI workflow for the project."""
        tasks = [
            WorkflowTask(
                id="setup",
                name="Setup Environment", 
                command="make install-dev",
                description="Install development dependencies",
                priority=TaskPriority.CRITICAL
            ),
            WorkflowTask(
                id="lint",
                name="Code Linting",
                command="make lint", 
                description="Run code quality checks",
                dependencies=["setup"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="type_check",
                name="Type Checking",
                command="make type-check",
                description="Run mypy type checking", 
                dependencies=["setup"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="security",
                name="Security Scan",
                command="make security",
                description="Run security vulnerability scans",
                dependencies=["setup"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="test_unit", 
                name="Unit Tests",
                command="make test-unit",
                description="Run unit test suite",
                dependencies=["lint", "type_check"],
                priority=TaskPriority.CRITICAL
            ),
            WorkflowTask(
                id="test_integration",
                name="Integration Tests", 
                command="make test-integration",
                description="Run integration test suite",
                dependencies=["test_unit"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="build",
                name="Build Package",
                command="make build",
                description="Build distribution package",
                dependencies=["test_unit", "security"],
                priority=TaskPriority.NORMAL
            )
        ]
        
        return self.create_workflow("standard_ci", tasks)
    
    def create_research_workflow(self) -> str:
        """Create research-focused workflow."""
        tasks = [
            WorkflowTask(
                id="setup_research",
                name="Setup Research Environment",
                command="make install-full",
                description="Install all dependencies including research tools",
                priority=TaskPriority.CRITICAL
            ),
            WorkflowTask(
                id="validate_baselines",
                name="Validate Baseline Models",
                command="python scripts/validate_baselines.py",
                description="Ensure baseline models are working",
                dependencies=["setup_research"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="run_experiments",
                name="Execute Experiments",
                command="python scripts/run_experiments.py",
                description="Run comparative studies and experiments",
                dependencies=["validate_baselines"],
                priority=TaskPriority.CRITICAL,
                timeout=1800  # 30 minutes
            ),
            WorkflowTask(
                id="analyze_results",
                name="Analyze Results",
                command="python scripts/analyze_results.py", 
                description="Statistical analysis of experimental results",
                dependencies=["run_experiments"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="generate_report",
                name="Generate Research Report",
                command="python scripts/generate_research_report.py",
                description="Create publication-ready research report",
                dependencies=["analyze_results"],
                priority=TaskPriority.NORMAL
            )
        ]
        
        return self.create_workflow("research", tasks)
    
    def shutdown(self):
        """Shutdown SDLC manager."""
        self.workflow_engine.shutdown()