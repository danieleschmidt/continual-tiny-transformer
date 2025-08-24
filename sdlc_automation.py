#!/usr/bin/env python3
"""
TERRAGON LABS - Autonomous SDLC Execution Framework
Advanced SDLC automation with self-healing, performance optimization, and intelligent monitoring.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sdlc_automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status tracking"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class QualityMetrics:
    """Quality gate metrics tracking"""
    test_coverage: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    code_quality_score: float = 0.0
    build_success: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AutomationTask:
    """SDLC automation task definition"""
    name: str
    command: str
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes default
    dependencies: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output: str = ""
    error: str = ""


class AutonomousSDLCExecutor:
    """
    Autonomous SDLC execution engine with progressive enhancement.
    Implements Generation 1-3 progressive enhancement strategy.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.metrics = QualityMetrics()
        self.tasks: Dict[str, AutomationTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize task definitions
        self._initialize_tasks()
        logger.info(f"Initialized Autonomous SDLC Executor for project: {self.project_root}")
    
    def _initialize_tasks(self):
        """Initialize SDLC automation tasks with progressive enhancement"""
        
        # Generation 1: Basic functionality
        generation_1_tasks = {
            "setup_environment": AutomationTask(
                name="Environment Setup",
                command="make install-dev",
                success_criteria={"exit_code": 0}
            ),
            "code_quality": AutomationTask(
                name="Code Quality Checks", 
                command="make lint && make type-check",
                dependencies=["setup_environment"],
                success_criteria={"exit_code": 0}
            ),
            "unit_tests": AutomationTask(
                name="Unit Tests",
                command="make test-unit",
                dependencies=["setup_environment"],
                success_criteria={"exit_code": 0, "coverage": 85.0}
            ),
            "build_validation": AutomationTask(
                name="Build Validation",
                command="make build",
                dependencies=["code_quality", "unit_tests"],
                success_criteria={"exit_code": 0}
            )
        }
        
        # Generation 2: Robust error handling and monitoring
        generation_2_tasks = {
            "security_scan": AutomationTask(
                name="Security Scan",
                command="make security",
                dependencies=["setup_environment"],
                success_criteria={"exit_code": 0, "vulnerabilities": 0}
            ),
            "integration_tests": AutomationTask(
                name="Integration Tests",
                command="make test-integration",
                dependencies=["unit_tests"],
                success_criteria={"exit_code": 0}
            ),
            "performance_tests": AutomationTask(
                name="Performance Tests", 
                command="make benchmark",
                dependencies=["build_validation"],
                success_criteria={"exit_code": 0, "regression_threshold": 10}
            ),
            "documentation_build": AutomationTask(
                name="Documentation Build",
                command="make docs",
                dependencies=["code_quality"],
                success_criteria={"exit_code": 0}
            )
        }
        
        # Generation 3: Performance optimization and scaling
        generation_3_tasks = {
            "load_testing": AutomationTask(
                name="Load Testing",
                command="python scripts/load_test.py",
                dependencies=["integration_tests"],
                success_criteria={"exit_code": 0, "response_time": 200}
            ),
            "container_validation": AutomationTask(
                name="Container Validation",
                command="make docker-build && make docker-run",
                dependencies=["build_validation"],
                success_criteria={"exit_code": 0}
            ),
            "deployment_readiness": AutomationTask(
                name="Deployment Readiness Check",
                command="python scripts/deployment_readiness.py",
                dependencies=["security_scan", "performance_tests"],
                success_criteria={"exit_code": 0, "readiness_score": 90}
            )
        }
        
        # Combine all generations
        self.tasks.update(generation_1_tasks)
        self.tasks.update(generation_2_tasks) 
        self.tasks.update(generation_3_tasks)
    
    async def execute_task(self, task_name: str) -> Tuple[bool, str]:
        """Execute a single SDLC task with error handling and retry logic"""
        task = self.tasks.get(task_name)
        if not task:
            return False, f"Task {task_name} not found"
        
        # Check dependencies
        for dep in task.dependencies:
            dep_task = self.tasks.get(dep)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False, f"Dependency {dep} not completed"
        
        task.status = TaskStatus.IN_PROGRESS
        task.start_time = datetime.now()
        
        try:
            logger.info(f"Starting task: {task.name}")
            
            # Execute with timeout
            process = await asyncio.create_subprocess_shell(
                task.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task.timeout
                )
                
                task.output = stdout.decode()
                task.error = stderr.decode()
                
                # Validate success criteria
                success = self._validate_success_criteria(task, process.returncode)
                
                if success:
                    task.status = TaskStatus.COMPLETED
                    task.end_time = datetime.now()
                    logger.info(f"✅ Task completed: {task.name}")
                    return True, task.output
                else:
                    raise Exception(f"Success criteria not met for {task.name}")
                    
            except asyncio.TimeoutError:
                process.kill()
                raise Exception(f"Task {task.name} timed out after {task.timeout}s")
                
        except Exception as e:
            error_msg = str(e)
            task.error = error_msg
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                logger.warning(f"🔄 Retrying task {task.name} (attempt {task.retry_count}/{task.max_retries})")
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                return await self.execute_task(task_name)
            else:
                task.status = TaskStatus.FAILED
                task.end_time = datetime.now()
                logger.error(f"❌ Task failed: {task.name} - {error_msg}")
                return False, error_msg
    
    def _validate_success_criteria(self, task: AutomationTask, exit_code: int) -> bool:
        """Validate task success criteria"""
        criteria = task.success_criteria
        
        # Check exit code
        if criteria.get("exit_code", 0) != exit_code:
            return False
        
        # Check coverage (extract from output)
        if "coverage" in criteria:
            coverage = self._extract_coverage_from_output(task.output)
            if coverage < criteria["coverage"]:
                return False
                
        # Additional criteria can be added here
        return True
    
    def _extract_coverage_from_output(self, output: str) -> float:
        """Extract test coverage percentage from output"""
        import re
        coverage_pattern = r'TOTAL.*?(\d+)%'
        match = re.search(coverage_pattern, output)
        return float(match.group(1)) if match else 0.0
    
    async def execute_generation(self, generation: int) -> Dict[str, bool]:
        """Execute a specific generation of tasks"""
        generation_tasks = {
            1: ["setup_environment", "code_quality", "unit_tests", "build_validation"],
            2: ["security_scan", "integration_tests", "performance_tests", "documentation_build"],
            3: ["load_testing", "container_validation", "deployment_readiness"]
        }
        
        tasks_to_run = generation_tasks.get(generation, [])
        results = {}
        
        logger.info(f"🚀 Starting Generation {generation} execution")
        
        for task_name in tasks_to_run:
            success, output = await self.execute_task(task_name)
            results[task_name] = success
            
            if not success:
                logger.error(f"Generation {generation} failed at task: {task_name}")
                return results
        
        logger.info(f"✅ Generation {generation} completed successfully")
        return results
    
    async def execute_full_sdlc(self) -> Dict[str, Any]:
        """Execute complete SDLC automation across all generations"""
        start_time = datetime.now()
        overall_results = {"generations": {}, "metrics": {}}
        
        logger.info("🎯 Starting Autonomous SDLC Execution")
        
        # Execute all generations progressively
        for generation in [1, 2, 3]:
            generation_results = await self.execute_generation(generation)
            overall_results["generations"][f"generation_{generation}"] = generation_results
            
            # If any generation fails, stop execution
            if not all(generation_results.values()):
                logger.error(f"SDLC execution failed at Generation {generation}")
                break
        
        # Collect final metrics
        overall_results["metrics"] = self._collect_final_metrics()
        overall_results["execution_time"] = (datetime.now() - start_time).total_seconds()
        overall_results["status"] = "completed" if self._all_critical_tasks_completed() else "failed"
        
        # Save results
        self._save_results(overall_results)
        
        return overall_results
    
    def _collect_final_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive quality metrics"""
        completed_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        total_tasks = len(self.tasks)
        
        return {
            "task_completion_rate": (completed_tasks / total_tasks) * 100,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED),
            "quality_gates_passed": self._count_quality_gates_passed()
        }
    
    def _count_quality_gates_passed(self) -> int:
        """Count quality gates that passed"""
        critical_tasks = ["code_quality", "unit_tests", "security_scan", "build_validation"]
        return sum(1 for task_name in critical_tasks 
                  if self.tasks.get(task_name, {}).status == TaskStatus.COMPLETED)
    
    def _all_critical_tasks_completed(self) -> bool:
        """Check if all critical tasks completed successfully"""
        critical_tasks = ["code_quality", "unit_tests", "security_scan", "build_validation"]
        return all(self.tasks.get(task_name, AutomationTask("", "")).status == TaskStatus.COMPLETED 
                  for task_name in critical_tasks)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save execution results to file"""
        results_file = self.project_root / "autonomous_sdlc_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📊 Results saved to: {results_file}")
    
    def get_execution_report(self) -> str:
        """Generate human-readable execution report"""
        report = []
        report.append("🎯 AUTONOMOUS SDLC EXECUTION REPORT")
        report.append("=" * 50)
        report.append(f"Project: {self.project_root.name}")
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        # Task status summary
        status_counts = {}
        for task in self.tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        
        report.append("📊 Task Summary:")
        for status, count in status_counts.items():
            report.append(f"  {status.title()}: {count}")
        report.append("")
        
        # Individual task details
        report.append("📋 Task Details:")
        for task_name, task in self.tasks.items():
            status_emoji = {"completed": "✅", "failed": "❌", "in_progress": "🔄", "pending": "⏳"}.get(task.status.value, "❓")
            duration = ""
            if task.start_time and task.end_time:
                duration = f" ({(task.end_time - task.start_time).total_seconds():.1f}s)"
            report.append(f"  {status_emoji} {task.name}{duration}")
            
            if task.error and task.status == TaskStatus.FAILED:
                report.append(f"    Error: {task.error[:100]}...")
        
        return "\n".join(report)


async def main():
    """Main execution entry point"""
    print("🚀 TERRAGON LABS - Autonomous SDLC Execution")
    print("=" * 60)
    
    executor = AutonomousSDLCExecutor()
    
    # Execute full SDLC autonomously
    results = await executor.execute_full_sdlc()
    
    # Display results
    print("\n" + executor.get_execution_report())
    
    # Exit with appropriate code
    sys.exit(0 if results["status"] == "completed" else 1)


if __name__ == "__main__":
    asyncio.run(main())