"""Automated SDLC workflow execution and task running."""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import threading
import queue
import schedule

from .core import WorkflowTask, WorkflowResult, WorkflowStatus, TaskPriority

logger = logging.getLogger(__name__)


class AutomationLevel(Enum):
    """Levels of automation for SDLC processes."""
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    FULLY_AUTO = "fully_auto"


class TriggerType(Enum):
    """Types of automation triggers."""
    MANUAL = "manual"
    FILE_CHANGE = "file_change"
    GIT_COMMIT = "git_commit"
    SCHEDULE = "schedule"
    API_CALL = "api_call"
    EXTERNAL_EVENT = "external_event"


@dataclass
class AutomationConfig:
    """Configuration for automated workflows."""
    level: AutomationLevel = AutomationLevel.SEMI_AUTO
    triggers: List[TriggerType] = None
    watch_patterns: List[str] = None
    schedule_cron: Optional[str] = None
    max_concurrent: int = 2
    retry_failed: bool = True
    notification_hooks: List[str] = None
    
    def __post_init__(self):
        if self.triggers is None:
            self.triggers = [TriggerType.MANUAL]
        if self.watch_patterns is None:
            self.watch_patterns = ["src/**/*.py", "tests/**/*.py"]
        if self.notification_hooks is None:
            self.notification_hooks = []


class TaskRunner:
    """Advanced task runner with automation capabilities."""
    
    def __init__(self, working_dir: str = None, config: AutomationConfig = None):
        self.working_dir = Path(working_dir or os.getcwd())
        self.config = config or AutomationConfig()
        self.task_queue = queue.Queue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_history = []
        
        # Threading for background execution
        self.executor_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # File monitoring
        self.file_watcher = None
        self.last_file_changes = {}
        
        # Scheduling
        self.scheduler_thread = None
        
        self.logger = logging.getLogger(f"{__name__}.TaskRunner")
    
    def start(self):
        """Start the task runner with background processing."""
        if self.running:
            self.logger.warning("Task runner already running")
            return
        
        self.running = True
        
        # Start task executor thread
        self.executor_thread = threading.Thread(target=self._task_executor, daemon=True)
        self.executor_thread.start()
        
        # Start file watcher if configured
        if TriggerType.FILE_CHANGE in self.config.triggers:
            self._start_file_watcher()
        
        # Start scheduler if configured
        if TriggerType.SCHEDULE in self.config.triggers and self.config.schedule_cron:
            self._start_scheduler()
        
        self.logger.info("Task runner started")
    
    def stop(self):
        """Stop the task runner."""
        self.running = False
        
        if self.executor_thread:
            self.executor_thread.join(timeout=5)
        
        if self.file_watcher:
            self._stop_file_watcher()
        
        if self.scheduler_thread:
            self._stop_scheduler()
        
        self.logger.info("Task runner stopped")
    
    def submit_task(self, task: WorkflowTask, trigger: TriggerType = TriggerType.MANUAL) -> str:
        """Submit a task for execution."""
        task_context = {
            'task': task,
            'trigger': trigger,
            'submitted_at': time.time(),
            'priority': task.priority.value
        }
        
        self.task_queue.put(task_context)
        self.logger.info(f"Submitted task {task.id} (trigger: {trigger.value})")
        
        return task.id
    
    def submit_workflow(self, tasks: List[WorkflowTask], trigger: TriggerType = TriggerType.MANUAL):
        """Submit multiple tasks as a workflow."""
        workflow_id = f"workflow_{int(time.time())}"
        
        for task in tasks:
            # Add workflow context to task
            task.environment = task.environment or {}
            task.environment['WORKFLOW_ID'] = workflow_id
            
            self.submit_task(task, trigger)
        
        self.logger.info(f"Submitted workflow {workflow_id} with {len(tasks)} tasks")
        return workflow_id
    
    def _task_executor(self):
        """Background task executor thread."""
        while self.running:
            try:
                # Get task with timeout
                try:
                    task_context = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Check concurrency limits
                with self.lock:
                    if len(self.running_tasks) >= self.config.max_concurrent:
                        # Put task back and wait
                        self.task_queue.put(task_context)
                        time.sleep(0.5)
                        continue
                
                # Execute task
                self._execute_task_context(task_context)
                
            except Exception as e:
                self.logger.error(f"Task executor error: {e}")
                time.sleep(1)
    
    def _execute_task_context(self, task_context: Dict):
        """Execute a task with full context tracking."""
        task = task_context['task']
        trigger = task_context['trigger']
        
        with self.lock:
            self.running_tasks[task.id] = {
                'task': task,
                'trigger': trigger,
                'start_time': time.time()
            }
        
        try:
            # Execute the actual task
            result = self._execute_single_task(task)
            
            # Update tracking
            with self.lock:
                del self.running_tasks[task.id]
                
                if result.status == WorkflowStatus.COMPLETED:
                    self.completed_tasks[task.id] = result
                else:
                    self.failed_tasks[task.id] = result
                    
                    # Retry logic
                    if self.config.retry_failed and result.exit_code != 0:
                        self.logger.info(f"Scheduling retry for failed task {task.id}")
                        # Implement exponential backoff retry
                        retry_task = task
                        retry_task.retry_count = max(1, retry_task.retry_count - 1)
                        if retry_task.retry_count > 0:
                            time.sleep(2)  # Simple delay for demo
                            self.submit_task(retry_task, TriggerType.MANUAL)
                
                # Add to history
                self.task_history.append({
                    'task_id': task.id,
                    'task_name': task.name,
                    'trigger': trigger.value,
                    'status': result.status.value,
                    'duration': result.duration,
                    'timestamp': time.time()
                })
            
            # Send notifications
            self._send_notifications(task, result, trigger)
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            
            with self.lock:
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
    
    def _execute_single_task(self, task: WorkflowTask) -> WorkflowResult:
        """Execute a single task and return result."""
        start_time = time.time()
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(task.environment)
            
            # Set working directory
            cwd = task.working_dir or str(self.working_dir)
            
            self.logger.info(f"Executing task: {task.name}")
            self.logger.debug(f"Command: {task.command}")
            self.logger.debug(f"Working dir: {cwd}")
            
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
            
            workflow_result = WorkflowResult(
                task_id=task.id,
                status=status,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                exit_code=result.returncode
            )
            
            if status == WorkflowStatus.COMPLETED:
                self.logger.info(f"Task {task.id} completed successfully in {duration:.2f}s")
            else:
                self.logger.error(f"Task {task.id} failed with exit code {result.returncode}")
            
            return workflow_result
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.error(f"Task {task.id} timed out after {task.timeout}s")
            
            return WorkflowResult(
                task_id=task.id,
                status=WorkflowStatus.FAILED,
                output="",
                error=f"Task timed out after {task.timeout} seconds",
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                exit_code=124  # Timeout exit code
            )
        
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.error(f"Task {task.id} execution error: {e}")
            
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
    
    def _start_file_watcher(self):
        """Start file system watcher for automatic triggers."""
        # Simple file change detection (would use watchdog in production)
        def check_file_changes():
            """Check for file changes and trigger workflows."""
            while self.running:
                try:
                    # Simple implementation: check modification times
                    for pattern in self.config.watch_patterns:
                        for file_path in self.working_dir.glob(pattern):
                            if file_path.is_file():
                                mtime = file_path.stat().st_mtime
                                
                                if str(file_path) not in self.last_file_changes:
                                    self.last_file_changes[str(file_path)] = mtime
                                elif mtime > self.last_file_changes[str(file_path)]:
                                    self.last_file_changes[str(file_path)] = mtime
                                    self._trigger_file_change_workflow(file_path)
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"File watcher error: {e}")
                    time.sleep(10)
        
        self.file_watcher = threading.Thread(target=check_file_changes, daemon=True)
        self.file_watcher.start()
        self.logger.info("File watcher started")
    
    def _stop_file_watcher(self):
        """Stop file system watcher."""
        if self.file_watcher:
            # File watcher will stop when self.running becomes False
            self.logger.info("File watcher stopped")
    
    def _trigger_file_change_workflow(self, changed_file: Path):
        """Trigger workflow based on file changes."""
        self.logger.info(f"File changed: {changed_file}")
        
        # Create a simple CI workflow for file changes
        ci_task = WorkflowTask(
            id=f"file_change_ci_{int(time.time())}",
            name="File Change CI",
            command="make ci",
            description=f"CI triggered by file change: {changed_file}",
            priority=TaskPriority.NORMAL,
            environment={'CHANGED_FILE': str(changed_file)}
        )
        
        self.submit_task(ci_task, TriggerType.FILE_CHANGE)
    
    def _start_scheduler(self):
        """Start scheduled workflow execution."""
        def scheduler_worker():
            while self.running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Scheduler error: {e}")
                    time.sleep(60)
        
        # Schedule based on cron expression (simplified)
        if self.config.schedule_cron:
            # For demo, run every hour (would parse cron properly in production)
            schedule.every().hour.do(self._trigger_scheduled_workflow)
        
        self.scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Scheduler started")
    
    def _stop_scheduler(self):
        """Stop scheduler."""
        if self.scheduler_thread:
            schedule.clear()
            self.logger.info("Scheduler stopped")
    
    def _trigger_scheduled_workflow(self):
        """Trigger scheduled workflow."""
        scheduled_task = WorkflowTask(
            id=f"scheduled_{int(time.time())}",
            name="Scheduled Workflow",
            command="make ci",
            description="Scheduled CI run",
            priority=TaskPriority.NORMAL
        )
        
        self.submit_task(scheduled_task, TriggerType.SCHEDULE)
    
    def _send_notifications(self, task: WorkflowTask, result: WorkflowResult, trigger: TriggerType):
        """Send notifications about task completion."""
        if not self.config.notification_hooks:
            return
        
        notification_data = {
            'task_id': task.id,
            'task_name': task.name,
            'status': result.status.value,
            'duration': result.duration,
            'trigger': trigger.value,
            'timestamp': time.time()
        }
        
        for hook in self.config.notification_hooks:
            try:
                # Simple webhook notification (would use proper HTTP client in production)
                self.logger.info(f"Sending notification to {hook}: {notification_data}")
            except Exception as e:
                self.logger.error(f"Failed to send notification to {hook}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current task runner status."""
        with self.lock:
            return {
                'running': self.running,
                'queued_tasks': self.task_queue.qsize(),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'total_history': len(self.task_history),
                'config': asdict(self.config)
            }
    
    def get_task_result(self, task_id: str) -> Optional[WorkflowResult]:
        """Get result for a specific task."""
        with self.lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            elif task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            return None


class AutomatedWorkflow:
    """High-level automated workflow orchestrator."""
    
    def __init__(self, project_path: str, config: AutomationConfig = None):
        self.project_path = Path(project_path)
        self.config = config or AutomationConfig()
        self.task_runner = TaskRunner(str(project_path), config)
        self.workflows = {}
        self.logger = logging.getLogger(f"{__name__}.AutomatedWorkflow")
    
    def start(self):
        """Start automated workflow system."""
        self.task_runner.start()
        self.logger.info("Automated workflow system started")
    
    def stop(self):
        """Stop automated workflow system."""
        self.task_runner.stop()
        self.logger.info("Automated workflow system stopped")
    
    def create_ci_workflow(self) -> str:
        """Create continuous integration workflow."""
        workflow_id = "ci_workflow"
        
        tasks = [
            WorkflowTask(
                id="ci_setup",
                name="CI Setup",
                command="make install-dev",
                description="Setup CI environment",
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="ci_lint", 
                name="CI Lint",
                command="make lint",
                description="Run linting checks",
                dependencies=["ci_setup"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="ci_test",
                name="CI Test",
                command="make test-coverage",
                description="Run comprehensive tests",
                dependencies=["ci_lint"],
                priority=TaskPriority.CRITICAL
            ),
            WorkflowTask(
                id="ci_security",
                name="CI Security",
                command="make security",
                description="Security vulnerability scan",
                dependencies=["ci_setup"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="ci_build",
                name="CI Build",
                command="make build",
                description="Build distribution package", 
                dependencies=["ci_test", "ci_security"],
                priority=TaskPriority.NORMAL
            )
        ]
        
        self.workflows[workflow_id] = {
            'name': 'Continuous Integration',
            'tasks': tasks,
            'created_at': time.time()
        }
        
        return workflow_id
    
    def create_deployment_workflow(self) -> str:
        """Create deployment workflow."""
        workflow_id = "deployment_workflow"
        
        tasks = [
            WorkflowTask(
                id="deploy_build",
                name="Build for Deployment",
                command="make build-clean",
                description="Clean build for deployment",
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="deploy_test",
                name="Pre-deployment Tests",
                command="make test-integration",
                description="Run integration tests before deployment",
                dependencies=["deploy_build"],
                priority=TaskPriority.CRITICAL
            ),
            WorkflowTask(
                id="deploy_package",
                name="Package Application",
                command="docker build -t continual-transformer:latest .",
                description="Build Docker image",
                dependencies=["deploy_test"],
                priority=TaskPriority.HIGH
            ),
            WorkflowTask(
                id="deploy_validate",
                name="Validate Deployment",
                command="python scripts/validate_deployment.py",
                description="Validate deployment configuration",
                dependencies=["deploy_package"],
                priority=TaskPriority.HIGH
            )
        ]
        
        self.workflows[workflow_id] = {
            'name': 'Deployment Pipeline',
            'tasks': tasks,
            'created_at': time.time()
        }
        
        return workflow_id
    
    def execute_workflow(self, workflow_id: str, trigger: TriggerType = TriggerType.MANUAL) -> str:
        """Execute a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = self.task_runner.submit_workflow(workflow['tasks'], trigger)
        
        self.logger.info(f"Executing workflow {workflow['name']} (trigger: {trigger.value})")
        return execution_id
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of workflow execution."""
        return self.task_runner.get_status()
    
    def enable_auto_ci(self):
        """Enable automatic CI on file changes."""
        if TriggerType.FILE_CHANGE not in self.config.triggers:
            self.config.triggers.append(TriggerType.FILE_CHANGE)
            self.task_runner.config = self.config
            self.task_runner._start_file_watcher()
        
        self.logger.info("Auto CI enabled on file changes")
    
    def schedule_nightly_build(self):
        """Schedule nightly build workflow."""
        if TriggerType.SCHEDULE not in self.config.triggers:
            self.config.triggers.append(TriggerType.SCHEDULE)
            self.config.schedule_cron = "0 2 * * *"  # 2 AM daily
            self.task_runner.config = self.config
            self.task_runner._start_scheduler()
        
        self.logger.info("Nightly build scheduled")