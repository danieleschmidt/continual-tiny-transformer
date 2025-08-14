"""Comprehensive tests for SDLC framework."""

import asyncio
import json
import os
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from continual_transformer.sdlc.core import (
    SDLCManager, WorkflowEngine, WorkflowTask, WorkflowStatus, TaskPriority
)
from continual_transformer.sdlc.automation import (
    AutomatedWorkflow, TaskRunner, AutomationConfig, TriggerType
)
from continual_transformer.sdlc.monitoring import (
    SDLCMonitor, MetricsCollector, SDLCMetrics
)
from continual_transformer.sdlc.reliability import (
    ReliabilityManager, CircuitBreaker, HealthChecker, FailureMode
)
from continual_transformer.sdlc.security import (
    SecurityValidator, SecurityLevel, SecretScanner, VulnerabilityLevel
)
from continual_transformer.sdlc.optimization import (
    OptimizedWorkflowEngine, OptimizationStrategy, ResourceMonitor, TaskProfiler
)


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create basic project structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "src" / "test_file.py").write_text("print('hello world')")
        
        yield project_path


@pytest.fixture
def sample_tasks():
    """Create sample workflow tasks for testing."""
    return [
        WorkflowTask(
            id="task1",
            name="Test Task 1",
            command="echo 'task 1'",
            description="First test task",
            priority=TaskPriority.HIGH
        ),
        WorkflowTask(
            id="task2",
            name="Test Task 2",
            command="echo 'task 2'",
            description="Second test task",
            dependencies=["task1"],
            priority=TaskPriority.NORMAL
        ),
        WorkflowTask(
            id="task3",
            name="Test Task 3",
            command="echo 'task 3'",
            description="Third test task",
            dependencies=["task1"],
            priority=TaskPriority.LOW
        )
    ]


@pytest.fixture
def failing_task():
    """Create a task that will fail."""
    return WorkflowTask(
        id="failing_task",
        name="Failing Task",
        command="exit 1",
        description="Task that always fails",
        priority=TaskPriority.NORMAL
    )


class TestWorkflowEngine:
    """Test the core workflow engine."""
    
    def test_workflow_engine_initialization(self):
        """Test workflow engine initializes correctly."""
        engine = WorkflowEngine(max_workers=2)
        assert engine.max_workers == 2
        assert engine.running_tasks == {}
        assert engine.completed_tasks == {}
        assert engine.failed_tasks == {}
    
    def test_single_task_execution_success(self, sample_tasks):
        """Test successful execution of a single task."""
        engine = WorkflowEngine()
        task = sample_tasks[0]  # Simple echo task
        
        result = engine.execute_task(task)
        
        assert result.status == WorkflowStatus.COMPLETED
        assert result.task_id == task.id
        assert result.exit_code == 0
        assert "task 1" in result.output
        assert result.duration > 0
    
    def test_single_task_execution_failure(self, failing_task):
        """Test handling of task failure."""
        engine = WorkflowEngine()
        
        result = engine.execute_task(failing_task)
        
        assert result.status == WorkflowStatus.FAILED
        assert result.task_id == failing_task.id
        assert result.exit_code == 1
        assert result.duration > 0
    
    def test_workflow_execution_with_dependencies(self, sample_tasks):
        """Test workflow execution respecting dependencies."""
        engine = WorkflowEngine()
        
        results = engine.execute_workflow(sample_tasks)
        
        assert len(results) == 3
        assert all(r.status == WorkflowStatus.COMPLETED for r in results.values())
        
        # Verify execution order was respected (task1 before task2 and task3)
        task1_end = results["task1"].end_time
        task2_start = results["task2"].start_time
        task3_start = results["task3"].start_time
        
        assert task1_end <= task2_start
        assert task1_end <= task3_start
    
    def test_workflow_engine_status(self, sample_tasks):
        """Test workflow engine status reporting."""
        engine = WorkflowEngine()
        
        # Initial status
        status = engine.get_status()
        assert status["running_tasks"] == 0
        assert status["completed_tasks"] == 0
        
        # Execute workflow
        results = engine.execute_workflow([sample_tasks[0]])
        
        # Status after execution
        status = engine.get_status()
        assert status["completed_tasks"] == 1
        assert status["total_results"] == 1


class TestSDLCManager:
    """Test the SDLC manager."""
    
    def test_sdlc_manager_initialization(self, temp_project_dir):
        """Test SDLC manager initializes correctly."""
        manager = SDLCManager(str(temp_project_dir))
        assert manager.project_path == temp_project_dir
        assert manager.workflows == {}
    
    def test_create_workflow(self, temp_project_dir, sample_tasks):
        """Test workflow creation."""
        manager = SDLCManager(str(temp_project_dir))
        
        workflow_id = manager.create_workflow("test_workflow", sample_tasks)
        
        assert workflow_id in manager.workflows
        workflow = manager.workflows[workflow_id]
        assert workflow["name"] == "test_workflow"
        assert len(workflow["tasks"]) == 3
        assert workflow["status"] == WorkflowStatus.PENDING
    
    def test_execute_workflow(self, temp_project_dir, sample_tasks):
        """Test workflow execution through manager."""
        manager = SDLCManager(str(temp_project_dir))
        
        workflow_id = manager.create_workflow("test_workflow", sample_tasks)
        results = manager.execute_workflow(workflow_id)
        
        assert len(results) == 3
        assert all(r.status == WorkflowStatus.COMPLETED for r in results.values())
        
        workflow = manager.workflows[workflow_id]
        assert workflow["status"] == WorkflowStatus.COMPLETED
    
    def test_create_standard_ci_workflow(self, temp_project_dir):
        """Test creation of standard CI workflow."""
        manager = SDLCManager(str(temp_project_dir))
        
        workflow_id = manager.create_standard_ci_workflow()
        
        assert workflow_id in manager.workflows
        workflow = manager.workflows[workflow_id]
        assert workflow["name"] == "standard_ci"
        
        # Verify CI tasks are present
        task_names = [task.name for task in workflow["tasks"]]
        expected_tasks = ["Setup Environment", "Code Linting", "Type Checking", "Security Scan", "Unit Tests"]
        for expected in expected_tasks:
            assert any(expected in name for name in task_names)


class TestTaskRunner:
    """Test the automated task runner."""
    
    def test_task_runner_initialization(self, temp_project_dir):
        """Test task runner initializes correctly."""
        runner = TaskRunner(str(temp_project_dir))
        assert runner.working_dir == temp_project_dir
        assert not runner.running
    
    def test_task_runner_start_stop(self, temp_project_dir):
        """Test starting and stopping task runner."""
        runner = TaskRunner(str(temp_project_dir))
        
        runner.start()
        assert runner.running
        
        runner.stop()
        assert not runner.running
    
    def test_submit_and_execute_task(self, temp_project_dir, sample_tasks):
        """Test submitting and executing tasks."""
        runner = TaskRunner(str(temp_project_dir))
        runner.start()
        
        try:
            task = sample_tasks[0]
            task_id = runner.submit_task(task)
            
            # Wait for task to complete
            time.sleep(2)
            
            result = runner.get_task_result(task_id)
            assert result is not None
            assert result.status == WorkflowStatus.COMPLETED
            
        finally:
            runner.stop()
    
    def test_task_runner_status(self, temp_project_dir):
        """Test task runner status reporting."""
        runner = TaskRunner(str(temp_project_dir))
        
        status = runner.get_status()
        assert "running" in status
        assert "queued_tasks" in status
        assert not status["running"]


class TestMetricsCollector:
    """Test the metrics collection system."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes correctly."""
        collector = MetricsCollector()
        assert collector.database_path == ":memory:"
        assert collector.workflow_metrics == {}
    
    def test_record_workflow_execution(self, sample_tasks):
        """Test recording workflow execution metrics."""
        collector = MetricsCollector()
        
        # Create mock results
        results = {}
        for task in sample_tasks:
            results[task.id] = Mock()
            results[task.id].status = WorkflowStatus.COMPLETED
            results[task.id].start_time = time.time() - 10
            results[task.id].end_time = time.time()
            results[task.id].duration = 5.0
        
        collector.record_workflow_execution(
            "test_workflow", "Test Workflow", results, "manual"
        )
        
        assert len(collector.execution_history) == 1
        execution = collector.execution_history[0]
        assert execution["workflow_id"] == "test_workflow"
        assert execution["success_count"] == 3
        assert execution["failure_count"] == 0
    
    def test_get_sdlc_metrics(self):
        """Test getting SDLC metrics."""
        collector = MetricsCollector()
        
        # Add some test data to database
        cursor = collector.conn.cursor()
        cursor.execute("""
            INSERT INTO workflow_executions 
            (workflow_id, workflow_name, status, start_time, end_time, duration, 
             task_count, success_count, failure_count, trigger_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "test_workflow", "Test", "completed", time.time() - 100, time.time(), 
            50.0, 3, 3, 0, "manual"
        ))
        collector.conn.commit()
        
        metrics = collector.get_sdlc_metrics(time_window_hours=1)
        
        assert isinstance(metrics, SDLCMetrics)
        assert metrics.workflow_count >= 1
        assert metrics.success_rate == 100.0


class TestReliabilityManager:
    """Test the reliability management system."""
    
    def test_reliability_manager_initialization(self, temp_project_dir):
        """Test reliability manager initializes correctly."""
        manager = ReliabilityManager(str(temp_project_dir))
        assert manager.project_path == temp_project_dir
        assert len(manager.failure_patterns) > 0
        assert isinstance(manager.health_checker, HealthChecker)
    
    def test_circuit_breaker_creation(self, temp_project_dir):
        """Test circuit breaker creation and management."""
        manager = ReliabilityManager(str(temp_project_dir))
        
        cb1 = manager.get_circuit_breaker("task1")
        cb2 = manager.get_circuit_breaker("task1")  # Same task
        cb3 = manager.get_circuit_breaker("task2")  # Different task
        
        assert cb1 is cb2  # Same instance for same task
        assert cb1 is not cb3  # Different instances for different tasks
        assert isinstance(cb1, CircuitBreaker)
    
    def test_failure_pattern_matching(self, temp_project_dir):
        """Test failure pattern matching."""
        manager = ReliabilityManager(str(temp_project_dir))
        
        # Create mock failed result
        result = Mock()
        result.status = WorkflowStatus.FAILED
        result.error = "permission denied accessing file"
        result.exit_code = 126
        
        pattern = manager.analyze_failure(result)
        
        assert pattern is not None
        assert pattern.failure_mode == FailureMode.PERMISSION_ERROR
    
    def test_health_checker(self, temp_project_dir):
        """Test health checking system."""
        manager = ReliabilityManager(str(temp_project_dir))
        
        health_report = manager.health_checker.get_health_report()
        
        assert "overall_healthy" in health_report
        assert "individual_checks" in health_report
        assert len(health_report["registered_checks"]) > 0
    
    def test_system_recovery(self, temp_project_dir):
        """Test system recovery functionality."""
        manager = ReliabilityManager(str(temp_project_dir))
        
        # Simulate some circuit breakers in open state
        cb = manager.get_circuit_breaker("test_task")
        with cb.lock:
            cb.state = "open"
            cb.failure_count = 5
        
        recovery_success = manager.perform_system_recovery()
        
        assert recovery_success
        assert cb.get_state()["state"] == "closed"
        assert cb.get_state()["failure_count"] == 0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes correctly."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_success_flow(self):
        """Test circuit breaker with successful calls."""
        cb = CircuitBreaker()
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_failure_flow(self):
        """Test circuit breaker with failing calls."""
        cb = CircuitBreaker(failure_threshold=2)
        
        def failing_func():
            raise Exception("test failure")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "closed"
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "open"
        assert cb.failure_count == 2
        
        # Next call should fail immediately due to open circuit
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(failing_func)


class TestSecurityValidator:
    """Test security validation system."""
    
    def test_security_validator_initialization(self, temp_project_dir):
        """Test security validator initializes correctly."""
        validator = SecurityValidator(str(temp_project_dir), SecurityLevel.STANDARD)
        assert validator.project_path == temp_project_dir
        assert validator.security_level == SecurityLevel.STANDARD
    
    def test_secret_scanner(self, temp_project_dir):
        """Test secret scanning functionality."""
        # Create file with potential secret
        test_file = temp_project_dir / "src" / "config.py"
        test_file.write_text("""
API_KEY = "abcd1234567890abcdef1234567890"
PASSWORD = "super_secret_password"
# This is a comment with API_KEY = "commented_key"
""")
        
        validator = SecurityValidator(str(temp_project_dir))
        vulnerabilities = validator.secret_scanner.scan_file(test_file)
        
        assert len(vulnerabilities) >= 2  # Should find API_KEY and PASSWORD
        
        # Check that commented secret has lower severity
        api_key_vulns = [v for v in vulnerabilities if "api" in v.title.lower()]
        assert any(v.severity == VulnerabilityLevel.HIGH for v in api_key_vulns)
    
    def test_comprehensive_security_scan(self, temp_project_dir):
        """Test comprehensive security scanning."""
        validator = SecurityValidator(str(temp_project_dir), SecurityLevel.BASIC)
        
        # This will run the scanners that are available
        result = validator.run_comprehensive_scan()
        
        assert result.scan_type == "comprehensive"
        assert result.duration > 0
        assert isinstance(result.vulnerabilities, list)
        assert isinstance(result.risk_score, float)
    
    def test_scan_validation(self, temp_project_dir):
        """Test security scan validation."""
        validator = SecurityValidator(str(temp_project_dir), SecurityLevel.STRICT)
        
        # Create mock scan result with high severity issues
        result = Mock()
        result.vulnerability_count_by_severity = {
            'critical': 1,
            'high': 2,
            'medium': 5,
            'low': 10
        }
        result.risk_score = 50
        
        is_valid, message = validator.validate_scan_result(result)
        
        assert not is_valid  # Should fail due to critical vulnerability
        assert "Critical vulnerabilities found" in message


class TestOptimizedWorkflowEngine:
    """Test the optimized workflow engine."""
    
    def test_optimized_engine_initialization(self):
        """Test optimized workflow engine initializes correctly."""
        engine = OptimizedWorkflowEngine(optimization_strategy=OptimizationStrategy.PARALLEL)
        assert engine.optimization_strategy == OptimizationStrategy.PARALLEL
        assert isinstance(engine.resource_monitor, ResourceMonitor)
        assert isinstance(engine.task_profiler, TaskProfiler)
    
    def test_resource_monitor(self):
        """Test resource monitoring functionality."""
        monitor = ResourceMonitor(monitoring_interval=0.1)
        
        monitor.start_monitoring()
        assert monitor.monitoring
        
        # Let it collect some metrics
        time.sleep(0.3)
        
        metrics = monitor.get_current_metrics()
        assert "timestamp" in metrics
        assert "system" in metrics
        
        availability = monitor.get_resource_availability()
        assert "cpu" in availability
        assert "memory" in availability
        assert all(0 <= v <= 1 for v in availability.values())
        
        monitor.stop_monitoring()
        assert not monitor.monitoring
    
    def test_task_profiler(self):
        """Test task profiling functionality."""
        profiler = TaskProfiler()
        
        # Create mock task and result
        task = Mock()
        task.id = "test_task"
        task.name = "Test Task"
        
        result = Mock()
        result.end_time = time.time()
        result.duration = 5.0
        result.status = WorkflowStatus.COMPLETED
        
        resource_metrics = {
            'cpu_percent': 50.0,
            'memory_mb': 100.0
        }
        
        # Record execution
        profiler.record_execution(task, result, resource_metrics)
        
        # Get profile
        profile = profiler.get_task_profile("test_task")
        assert profile is not None
        assert profile.task_id == "test_task"
        assert profile.avg_duration_seconds == 5.0
    
    @patch('subprocess.run')
    def test_optimized_workflow_execution(self, mock_subprocess, sample_tasks):
        """Test optimized workflow execution."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        engine = OptimizedWorkflowEngine(optimization_strategy=OptimizationStrategy.SEQUENTIAL)
        
        try:
            results = engine.execute_optimized_workflow(sample_tasks)
            
            assert len(results) == 3
            assert all(r.status == WorkflowStatus.COMPLETED for r in results.values())
            
            # Check performance metrics
            metrics = engine.get_performance_metrics()
            assert "total_workflows" in metrics
            assert metrics["total_workflows"] >= 1
            
        finally:
            engine.shutdown()


class TestAutomatedWorkflow:
    """Test automated workflow system."""
    
    def test_automated_workflow_initialization(self, temp_project_dir):
        """Test automated workflow system initializes correctly."""
        workflow = AutomatedWorkflow(str(temp_project_dir))
        assert workflow.project_path == temp_project_dir
        assert isinstance(workflow.task_runner, TaskRunner)
    
    def test_create_ci_workflow(self, temp_project_dir):
        """Test CI workflow creation."""
        workflow = AutomatedWorkflow(str(temp_project_dir))
        
        workflow_id = workflow.create_ci_workflow()
        assert workflow_id == "ci_workflow"
        assert workflow_id in workflow.workflows
        
        ci_workflow = workflow.workflows[workflow_id]
        assert ci_workflow["name"] == "Continuous Integration"
        assert len(ci_workflow["tasks"]) > 0
    
    def test_create_deployment_workflow(self, temp_project_dir):
        """Test deployment workflow creation."""
        workflow = AutomatedWorkflow(str(temp_project_dir))
        
        workflow_id = workflow.create_deployment_workflow()
        assert workflow_id == "deployment_workflow"
        assert workflow_id in workflow.workflows
        
        deploy_workflow = workflow.workflows[workflow_id]
        assert deploy_workflow["name"] == "Deployment Pipeline"
        assert len(deploy_workflow["tasks"]) > 0


class TestSDLCMonitor:
    """Test SDLC monitoring system."""
    
    def test_sdlc_monitor_initialization(self, temp_project_dir):
        """Test SDLC monitor initializes correctly."""
        monitor = SDLCMonitor(str(temp_project_dir))
        assert monitor.project_path == temp_project_dir
        assert isinstance(monitor.metrics_collector, MetricsCollector)
        assert not monitor.monitoring_active
    
    def test_start_stop_monitoring(self, temp_project_dir):
        """Test starting and stopping monitoring."""
        monitor = SDLCMonitor(str(temp_project_dir))
        
        monitor.start_monitoring(interval_seconds=0.1)
        assert monitor.monitoring_active
        
        # Let it run briefly
        time.sleep(0.3)
        
        monitor.stop_monitoring()
        assert not monitor.monitoring_active
    
    def test_dashboard_data(self, temp_project_dir):
        """Test dashboard data generation."""
        monitor = SDLCMonitor(str(temp_project_dir))
        
        dashboard_data = monitor.get_dashboard_data()
        
        assert "metrics" in dashboard_data
        assert "system_status" in dashboard_data
        assert "monitoring_active" in dashboard_data["system_status"]


# Integration Tests
class TestSDLCIntegration:
    """Integration tests for the complete SDLC system."""
    
    def test_end_to_end_workflow_execution(self, temp_project_dir, sample_tasks):
        """Test complete end-to-end workflow execution."""
        # Initialize all components
        manager = SDLCManager(str(temp_project_dir))
        monitor = SDLCMonitor(str(temp_project_dir))
        reliability_manager = ReliabilityManager(str(temp_project_dir))
        
        try:
            # Start monitoring
            monitor.start_monitoring(interval_seconds=0.5)
            
            # Create and execute workflow
            workflow_id = manager.create_workflow("integration_test", sample_tasks)
            results = manager.execute_workflow(workflow_id)
            
            # Verify execution
            assert len(results) == 3
            assert all(r.status == WorkflowStatus.COMPLETED for r in results.values())
            
            # Check workflow status
            workflow_status = manager.get_workflow_status(workflow_id)
            assert workflow_status["status"] == WorkflowStatus.COMPLETED
            
            # Check system health
            health_report = reliability_manager.health_checker.get_health_report()
            assert "overall_healthy" in health_report
            
            # Let monitoring collect some data
            time.sleep(1)
            
            # Get dashboard data
            dashboard_data = monitor.get_dashboard_data()
            assert dashboard_data["system_status"]["monitoring_active"]
            
        finally:
            monitor.stop_monitoring()
    
    def test_failure_recovery_integration(self, temp_project_dir, failing_task):
        """Test failure recovery integration."""
        manager = SDLCManager(str(temp_project_dir))
        reliability_manager = ReliabilityManager(str(temp_project_dir))
        
        # Create workflow with failing task
        workflow_id = manager.create_workflow("failure_test", [failing_task])
        results = manager.execute_workflow(workflow_id)
        
        # Verify failure was detected
        assert results[failing_task.id].status == WorkflowStatus.FAILED
        
        # Test recovery
        recovery_success = reliability_manager.perform_system_recovery()
        assert recovery_success
        
        # Get reliability report
        report = reliability_manager.get_reliability_report()
        assert "system_health" in report
        assert "reliability_score" in report
    
    @patch('subprocess.run')
    def test_optimized_workflow_with_monitoring(self, mock_subprocess, temp_project_dir, sample_tasks):
        """Test optimized workflow with monitoring integration."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        engine = OptimizedWorkflowEngine(optimization_strategy=OptimizationStrategy.INTELLIGENT)
        monitor = SDLCMonitor(str(temp_project_dir))
        
        try:
            monitor.start_monitoring(interval_seconds=0.1)
            
            results = engine.execute_optimized_workflow(sample_tasks)
            
            assert len(results) == 3
            assert all(r.status == WorkflowStatus.COMPLETED for r in results.values())
            
            # Check performance metrics
            performance_metrics = engine.get_performance_metrics()
            assert performance_metrics["total_workflows"] >= 1
            
            # Let monitoring collect data
            time.sleep(0.5)
            
            dashboard_data = monitor.get_dashboard_data()
            assert dashboard_data["system_status"]["monitoring_active"]
            
        finally:
            engine.shutdown()
            monitor.stop_monitoring()


# Performance Tests
class TestSDLCPerformance:
    """Performance tests for SDLC components."""
    
    def test_workflow_engine_performance(self):
        """Test workflow engine performance with multiple tasks."""
        engine = WorkflowEngine(max_workers=4)
        
        # Create many simple tasks
        tasks = []
        for i in range(20):
            tasks.append(WorkflowTask(
                id=f"perf_task_{i}",
                name=f"Performance Task {i}",
                command="echo 'performance test'",
                description=f"Performance test task {i}",
                priority=TaskPriority.NORMAL
            ))
        
        start_time = time.time()
        results = engine.execute_workflow(tasks)
        execution_time = time.time() - start_time
        
        assert len(results) == 20
        assert all(r.status == WorkflowStatus.COMPLETED for r in results.values())
        assert execution_time < 30  # Should complete within 30 seconds
    
    def test_metrics_collector_performance(self):
        """Test metrics collector performance with high volume data."""
        collector = MetricsCollector()
        
        # Generate many workflow executions
        start_time = time.time()
        for i in range(100):
            results = {}
            for j in range(5):
                result = Mock()
                result.status = WorkflowStatus.COMPLETED
                result.start_time = time.time() - 10
                result.end_time = time.time()
                result.duration = 2.0
                results[f"task_{j}"] = result
            
            collector.record_workflow_execution(
                f"workflow_{i}", f"Test Workflow {i}", results, "test"
            )
        
        collection_time = time.time() - start_time
        
        # Get metrics
        metrics = collector.get_sdlc_metrics()
        assert metrics.workflow_count >= 100
        
        # Cleanup performance
        cleanup_start = time.time()
        collector.cleanup(days_to_keep=0)  # Delete all
        cleanup_time = time.time() - cleanup_start
        
        assert collection_time < 10  # Should collect within 10 seconds
        assert cleanup_time < 5   # Should cleanup within 5 seconds
        
        collector.close()


# Stress Tests
class TestSDLCStress:
    """Stress tests for SDLC system resilience."""
    
    def test_concurrent_workflow_execution(self, temp_project_dir):
        """Test concurrent workflow execution."""
        manager = SDLCManager(str(temp_project_dir))
        
        def execute_workflow(workflow_num):
            tasks = [
                WorkflowTask(
                    id=f"stress_task_{workflow_num}_{i}",
                    name=f"Stress Task {workflow_num}-{i}",
                    command=f"echo 'stress test {workflow_num}-{i}'",
                    description=f"Stress test task {workflow_num}-{i}"
                ) for i in range(3)
            ]
            
            workflow_id = manager.create_workflow(f"stress_workflow_{workflow_num}", tasks)
            return manager.execute_workflow(workflow_id)
        
        # Execute multiple workflows concurrently
        import threading
        results = {}
        threads = []
        
        for i in range(5):
            thread = threading.Thread(
                target=lambda i=i: results.update({f"workflow_{i}": execute_workflow(i)})
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        assert len(results) == 5
        for workflow_results in results.values():
            assert len(workflow_results) == 3
            assert all(r.status == WorkflowStatus.COMPLETED for r in workflow_results.values())
    
    def test_resource_exhaustion_handling(self, temp_project_dir):
        """Test handling of resource exhaustion scenarios."""
        # This test simulates resource exhaustion by creating memory-intensive tasks
        engine = OptimizedWorkflowEngine(max_workers=2)  # Limited workers
        
        try:
            # Create tasks that simulate high resource usage
            tasks = []
            for i in range(10):  # More tasks than workers
                tasks.append(WorkflowTask(
                    id=f"resource_task_{i}",
                    name=f"Resource Task {i}",
                    command="sleep 0.1 && echo 'resource test'",  # Short but multiple
                    description=f"Resource intensive task {i}",
                    timeout=5  # Short timeout to prevent hanging
                ))
            
            start_time = time.time()
            results = engine.execute_optimized_workflow(tasks)
            execution_time = time.time() - start_time
            
            assert len(results) == 10
            # Some tasks might fail due to resource constraints, but system should handle gracefully
            completed_count = sum(1 for r in results.values() if r.status == WorkflowStatus.COMPLETED)
            assert completed_count >= 5  # At least half should complete
            assert execution_time < 30  # Should not hang indefinitely
            
        finally:
            engine.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])