"""Integration tests for SDLC components with real scenarios."""

import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from continual_transformer.sdlc.core import SDLCManager, WorkflowTask, TaskPriority
from continual_transformer.sdlc.automation import AutomatedWorkflow, AutomationConfig, TriggerType
from continual_transformer.sdlc.monitoring import SDLCMonitor
from continual_transformer.sdlc.reliability import ReliabilityManager
from continual_transformer.sdlc.security import SecurityValidator, SecurityLevel
from continual_transformer.sdlc.optimization import OptimizedWorkflowEngine, OptimizationStrategy


@pytest.fixture
def ml_project_structure():
    """Create realistic ML project structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create realistic project structure
        (project_path / "src" / "continual_transformer").mkdir(parents=True)
        (project_path / "tests").mkdir()
        (project_path / "docs").mkdir()
        (project_path / "scripts").mkdir()
        (project_path / "data").mkdir()
        (project_path / "models").mkdir()
        
        # Create Python files
        (project_path / "src" / "continual_transformer" / "__init__.py").write_text("")
        (project_path / "src" / "continual_transformer" / "model.py").write_text("""
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
""")
        
        (project_path / "tests" / "test_model.py").write_text("""
import pytest
import torch
from src.continual_transformer.model import SimpleModel

def test_model_creation():
    model = SimpleModel()
    assert model is not None

def test_model_forward():
    model = SimpleModel()
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 1)
""")
        
        # Create requirements file
        (project_path / "requirements.txt").write_text("""
torch>=1.12.0
numpy>=1.21.0
pytest>=7.0.0
""")
        
        # Create setup.py
        (project_path / "setup.py").write_text("""
from setuptools import setup, find_packages

setup(
    name="continual-transformer-test",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
    ],
)
""")
        
        # Create Makefile
        (project_path / "Makefile").write_text("""
test:
\tpython -m pytest tests/

lint:
\techo "Running linting..." && echo "Linting completed"

install:
\tpip install -e .

build:
\tpython setup.py sdist bdist_wheel

clean:
\trm -rf build/ dist/ *.egg-info/
""")
        
        # Create config file with potential secrets (for security testing)
        (project_path / "config.py").write_text("""
# Configuration file
API_KEY = "sk-test-1234567890abcdef1234567890abcdef"
DATABASE_URL = "postgresql://user:password@localhost:5432/db"
DEBUG = True

# Safe configuration
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
""")
        
        yield project_path


class TestRealWorldMLWorkflow:
    """Test realistic ML development workflows."""
    
    def test_complete_ml_development_cycle(self, ml_project_structure):
        """Test a complete ML development cycle with all SDLC components."""
        project_path = ml_project_structure
        
        # Initialize SDLC components
        sdlc_manager = SDLCManager(str(project_path))
        monitor = SDLCMonitor(str(project_path))
        reliability_manager = ReliabilityManager(str(project_path))
        
        try:
            # Start monitoring
            monitor.start_monitoring(interval_seconds=1)
            
            # Create ML development workflow
            ml_tasks = [
                WorkflowTask(
                    id="setup_env",
                    name="Setup Environment",
                    command="echo 'Setting up Python environment'",
                    description="Setup Python virtual environment",
                    priority=TaskPriority.CRITICAL
                ),
                WorkflowTask(
                    id="install_deps",
                    name="Install Dependencies",
                    command="echo 'Installing dependencies'",
                    description="Install ML dependencies",
                    dependencies=["setup_env"],
                    priority=TaskPriority.HIGH
                ),
                WorkflowTask(
                    id="run_tests",
                    name="Run Tests",
                    command="echo 'Running tests...' && sleep 1 && echo 'Tests passed'",
                    description="Run ML model tests",
                    dependencies=["install_deps"],
                    priority=TaskPriority.HIGH
                ),
                WorkflowTask(
                    id="train_model",
                    name="Train Model",
                    command="echo 'Training ML model...' && sleep 2 && echo 'Model trained'",
                    description="Train continual learning model",
                    dependencies=["run_tests"],
                    priority=TaskPriority.NORMAL
                ),
                WorkflowTask(
                    id="validate_model",
                    name="Validate Model",
                    command="echo 'Validating model performance...' && sleep 1 && echo 'Validation complete'",
                    description="Validate model performance",
                    dependencies=["train_model"],
                    priority=TaskPriority.HIGH
                )
            ]
            
            # Create and execute workflow
            workflow_id = sdlc_manager.create_workflow("ml_development", ml_tasks)
            results = sdlc_manager.execute_workflow(workflow_id)
            
            # Verify all tasks completed successfully
            assert len(results) == 5
            for task_id, result in results.items():
                assert result.status.value == "completed", f"Task {task_id} failed: {result.error}"
            
            # Verify execution order (dependencies respected)
            setup_end = results["setup_env"].end_time
            install_start = results["install_deps"].start_time
            assert setup_end <= install_start, "Dependencies not respected"
            
            # Check workflow status
            workflow_status = sdlc_manager.get_workflow_status(workflow_id)
            assert workflow_status["status"].value == "completed"
            
            # Give monitoring time to collect data
            time.sleep(2)
            
            # Verify monitoring collected data
            dashboard_data = monitor.get_dashboard_data()
            assert dashboard_data["system_status"]["monitoring_active"]
            
            # Check reliability report
            reliability_report = reliability_manager.get_reliability_report()
            assert "system_health" in reliability_report
            assert reliability_report["reliability_score"] > 50  # Should be healthy
            
        finally:
            monitor.stop_monitoring()
    
    def test_ci_cd_pipeline_simulation(self, ml_project_structure):
        """Test CI/CD pipeline simulation for ML project."""
        project_path = ml_project_structure
        
        # Use automated workflow system
        automation_config = AutomationConfig(
            level="semi_auto",
            triggers=[TriggerType.MANUAL],
            max_concurrent=2
        )
        
        automated_workflow = AutomatedWorkflow(str(project_path), automation_config)
        
        try:
            automated_workflow.start()
            
            # Create CI workflow
            ci_workflow_id = automated_workflow.create_ci_workflow()
            
            # Execute CI workflow
            execution_id = automated_workflow.execute_workflow(ci_workflow_id)
            
            # Wait for execution to complete
            time.sleep(5)
            
            # Check workflow status
            status = automated_workflow.get_workflow_status(ci_workflow_id)
            assert "running" in status
            
            # Create deployment workflow
            deploy_workflow_id = automated_workflow.create_deployment_workflow()
            
            # Verify workflows were created
            assert ci_workflow_id in automated_workflow.workflows
            assert deploy_workflow_id in automated_workflow.workflows
            
        finally:
            automated_workflow.stop()
    
    def test_security_scanning_integration(self, ml_project_structure):
        """Test security scanning on realistic ML project."""
        project_path = ml_project_structure
        
        # Initialize security validator with strict settings
        security_validator = SecurityValidator(str(project_path), SecurityLevel.STRICT)
        
        # Run comprehensive security scan
        scan_result = security_validator.run_comprehensive_scan()
        
        # Verify scan executed
        assert scan_result.scan_type == "comprehensive"
        assert scan_result.duration > 0
        assert scan_result.files_scanned > 0
        
        # Should find the secrets we planted in config.py
        vulnerabilities = scan_result.vulnerabilities
        assert len(vulnerabilities) >= 2  # API_KEY and DATABASE_URL
        
        # Verify secret detection
        secret_vulns = [v for v in vulnerabilities if "secret" in v.title.lower() or "api" in v.title.lower()]
        assert len(secret_vulns) >= 1
        
        # Check validation against security policy
        is_valid, message = security_validator.validate_scan_result(scan_result)
        # Should fail due to strict security level and planted secrets
        assert not is_valid or "vulnerabilities found" in message
        
        # Generate security report
        report_path = project_path / "security_report.json"
        security_validator.generate_security_report(scan_result, str(report_path))
        assert report_path.exists()
    
    @patch('subprocess.run')
    def test_optimized_ml_training_pipeline(self, mock_subprocess, ml_project_structure):
        """Test optimized execution of ML training pipeline."""
        project_path = ml_project_structure
        
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Create optimized workflow engine
        engine = OptimizedWorkflowEngine(
            max_workers=4,
            optimization_strategy=OptimizationStrategy.INTELLIGENT
        )
        
        try:
            # Create ML training tasks with different resource profiles
            training_tasks = [
                WorkflowTask(
                    id="data_preprocessing",
                    name="Data Preprocessing",
                    command="python scripts/preprocess_data.py",
                    description="Preprocess training data",
                    priority=TaskPriority.HIGH,
                    timeout=300
                ),
                WorkflowTask(
                    id="feature_engineering",
                    name="Feature Engineering", 
                    command="python scripts/feature_engineering.py",
                    description="Engineer features for training",
                    dependencies=["data_preprocessing"],
                    priority=TaskPriority.HIGH,
                    timeout=600
                ),
                WorkflowTask(
                    id="model_training",
                    name="Model Training",
                    command="python scripts/train_model.py --epochs 10",
                    description="Train continual learning model",
                    dependencies=["feature_engineering"],
                    priority=TaskPriority.CRITICAL,
                    timeout=3600
                ),
                WorkflowTask(
                    id="model_evaluation",
                    name="Model Evaluation",
                    command="python scripts/evaluate_model.py",
                    description="Evaluate trained model",
                    dependencies=["model_training"],
                    priority=TaskPriority.HIGH,
                    timeout=300
                ),
                WorkflowTask(
                    id="generate_report",
                    name="Generate Training Report",
                    command="python scripts/generate_report.py",
                    description="Generate training results report",
                    dependencies=["model_evaluation"],
                    priority=TaskPriority.NORMAL,
                    timeout=120
                )
            ]
            
            # Execute optimized workflow
            results = engine.execute_optimized_workflow(training_tasks)
            
            # Verify execution
            assert len(results) == 5
            assert all(r.status.value == "completed" for r in results.values())
            
            # Verify optimization was applied
            performance_metrics = engine.get_performance_metrics()
            assert performance_metrics["total_workflows"] >= 1
            assert "optimization_strategies_used" in performance_metrics
            
        finally:
            engine.shutdown()


class TestFailureRecoveryScenarios:
    """Test various failure scenarios and recovery mechanisms."""
    
    def test_dependency_failure_recovery(self, ml_project_structure):
        """Test recovery from dependency installation failures."""
        project_path = ml_project_structure
        
        sdlc_manager = SDLCManager(str(project_path))
        reliability_manager = ReliabilityManager(str(project_path))
        
        # Create workflow with failing dependency installation
        tasks = [
            WorkflowTask(
                id="install_deps",
                name="Install Dependencies",
                command="exit 1",  # Simulated failure
                description="Install project dependencies",
                priority=TaskPriority.CRITICAL
            ),
            WorkflowTask(
                id="run_tests",
                name="Run Tests",
                command="echo 'Running tests'",
                description="Run test suite",
                dependencies=["install_deps"],
                priority=TaskPriority.HIGH
            )
        ]
        
        # Execute workflow (should fail)
        workflow_id = sdlc_manager.create_workflow("dependency_test", tasks)
        results = sdlc_manager.execute_workflow(workflow_id)
        
        # Verify failure was captured
        assert results["install_deps"].status.value == "failed"
        
        # Test failure analysis
        failure_result = results["install_deps"]
        failure_pattern = reliability_manager.analyze_failure(failure_result)
        assert failure_pattern is not None
        
        # Test recovery attempt
        recovery_success, recovery_task = reliability_manager.attempt_recovery(
            tasks[0], failure_result, attempt_number=1
        )
        
        # Recovery behavior depends on the pattern matched
        assert isinstance(recovery_success, bool)
    
    def test_network_failure_simulation(self, ml_project_structure):
        """Test handling of network-related failures."""
        project_path = ml_project_structure
        
        reliability_manager = ReliabilityManager(str(project_path))
        
        # Create mock network failure result
        network_failure_result = Mock()
        network_failure_result.status.value = "failed"
        network_failure_result.error = "Connection timeout: unable to reach repository"
        network_failure_result.exit_code = 1
        
        # Test failure pattern matching
        pattern = reliability_manager.analyze_failure(network_failure_result)
        assert pattern is not None
        assert pattern.failure_mode.value == "network_error" or pattern.failure_mode.value == "unknown_error"
    
    def test_resource_exhaustion_handling(self, ml_project_structure):
        """Test handling of resource exhaustion scenarios."""
        project_path = ml_project_structure
        
        # Create optimized engine with limited resources
        engine = OptimizedWorkflowEngine(max_workers=1)  # Very limited
        
        try:
            # Create many concurrent tasks to stress the system
            stress_tasks = []
            for i in range(20):
                stress_tasks.append(WorkflowTask(
                    id=f"stress_task_{i}",
                    name=f"Stress Task {i}",
                    command="echo 'processing' && sleep 0.1",
                    description=f"Stress test task {i}",
                    timeout=5
                ))
            
            start_time = time.time()
            with patch('subprocess.run') as mock_run:
                # Mock successful execution but with delay
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "success"
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                results = engine.execute_optimized_workflow(stress_tasks)
                
            execution_time = time.time() - start_time
            
            # Verify system handled the load gracefully
            assert len(results) == 20
            completed_count = sum(1 for r in results.values() if r.status.value == "completed")
            assert completed_count >= 15  # Most should complete
            assert execution_time < 30  # Should not take too long
            
        finally:
            engine.shutdown()


class TestPerformanceOptimization:
    """Test performance optimization scenarios."""
    
    @patch('subprocess.run')
    def test_parallel_vs_sequential_performance(self, mock_subprocess, ml_project_structure):
        """Compare parallel vs sequential execution performance."""
        project_path = ml_project_structure
        
        # Mock subprocess to simulate work
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "task completed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Create tasks without dependencies (can be parallelized)
        parallel_tasks = []
        for i in range(8):
            parallel_tasks.append(WorkflowTask(
                id=f"parallel_task_{i}",
                name=f"Parallel Task {i}",
                command=f"echo 'task {i}' && sleep 0.1",
                description=f"Parallelizable task {i}",
                timeout=10
            ))
        
        # Test sequential execution
        sequential_engine = OptimizedWorkflowEngine(
            optimization_strategy=OptimizationStrategy.SEQUENTIAL
        )
        
        try:
            start_time = time.time()
            sequential_results = sequential_engine.execute_optimized_workflow(parallel_tasks)
            sequential_time = time.time() - start_time
            
            assert all(r.status.value == "completed" for r in sequential_results.values())
            
        finally:
            sequential_engine.shutdown()
        
        # Test parallel execution
        parallel_engine = OptimizedWorkflowEngine(
            optimization_strategy=OptimizationStrategy.PARALLEL,
            max_workers=4
        )
        
        try:
            start_time = time.time()
            parallel_results = parallel_engine.execute_optimized_workflow(parallel_tasks)
            parallel_time = time.time() - start_time
            
            assert all(r.status.value == "completed" for r in parallel_results.values())
            
        finally:
            parallel_engine.shutdown()
        
        # Parallel should be faster (though mocked, the framework overhead differs)
        print(f"Sequential time: {sequential_time:.2f}s, Parallel time: {parallel_time:.2f}s")
        # Note: With mocking, timing differences might be minimal, but structure should be correct
    
    def test_resource_aware_optimization(self, ml_project_structure):
        """Test resource-aware task optimization."""
        project_path = ml_project_structure
        
        engine = OptimizedWorkflowEngine(
            optimization_strategy=OptimizationStrategy.RESOURCE_AWARE
        )
        
        try:
            # Let the resource monitor collect some baseline data
            time.sleep(1)
            
            # Create tasks with different resource profiles
            resource_tasks = [
                WorkflowTask(
                    id="cpu_intensive",
                    name="CPU Intensive Task",
                    command="echo 'CPU intensive work'",
                    description="Simulated CPU-bound task"
                ),
                WorkflowTask(
                    id="io_intensive",
                    name="I/O Intensive Task", 
                    command="echo 'I/O intensive work'",
                    description="Simulated I/O-bound task"
                ),
                WorkflowTask(
                    id="memory_intensive",
                    name="Memory Intensive Task",
                    command="echo 'Memory intensive work'",
                    description="Simulated memory-bound task"
                )
            ]
            
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "completed"
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                results = engine.execute_optimized_workflow(resource_tasks)
                
            assert len(results) == 3
            assert all(r.status.value == "completed" for r in results.values())
            
            # Verify optimization was applied
            performance_metrics = engine.get_performance_metrics()
            assert "optimization_strategies_used" in performance_metrics
            
        finally:
            engine.shutdown()


class TestMonitoringAndMetrics:
    """Test monitoring and metrics collection in realistic scenarios."""
    
    def test_long_running_workflow_monitoring(self, ml_project_structure):
        """Test monitoring of long-running ML workflows."""
        project_path = ml_project_structure
        
        monitor = SDLCMonitor(str(project_path))
        sdlc_manager = SDLCManager(str(project_path))
        
        try:
            # Start monitoring with high frequency for testing
            monitor.start_monitoring(interval_seconds=0.5)
            
            # Create long-running workflow
            long_tasks = [
                WorkflowTask(
                    id="long_task_1",
                    name="Long Running Task 1",
                    command="echo 'Starting long task 1' && sleep 2 && echo 'Completed'",
                    description="Simulated long-running task 1"
                ),
                WorkflowTask(
                    id="long_task_2", 
                    name="Long Running Task 2",
                    command="echo 'Starting long task 2' && sleep 1 && echo 'Completed'",
                    description="Simulated long-running task 2",
                    dependencies=["long_task_1"]
                )
            ]
            
            workflow_id = sdlc_manager.create_workflow("long_workflow", long_tasks)
            results = sdlc_manager.execute_workflow(workflow_id)
            
            # Let monitoring collect data during execution
            time.sleep(1)
            
            # Verify execution
            assert all(r.status.value == "completed" for r in results.values())
            
            # Check that monitoring collected meaningful data
            dashboard_data = monitor.get_dashboard_data()
            assert dashboard_data["system_status"]["monitoring_active"]
            
            # Verify metrics were collected
            metrics = dashboard_data["metrics"]
            assert "workflow_count" in metrics
            
        finally:
            monitor.stop_monitoring()
    
    def test_metrics_export_and_analysis(self, ml_project_structure):
        """Test metrics export and analysis capabilities."""
        project_path = ml_project_structure
        
        monitor = SDLCMonitor(str(project_path))
        
        try:
            # Execute several workflows to generate metrics data
            sdlc_manager = SDLCManager(str(project_path))
            
            for i in range(3):
                tasks = [
                    WorkflowTask(
                        id=f"metrics_task_{i}_{j}",
                        name=f"Metrics Task {i}-{j}",
                        command=f"echo 'metrics test {i}-{j}'",
                        description=f"Task for metrics testing {i}-{j}"
                    ) for j in range(2)
                ]
                
                workflow_id = sdlc_manager.create_workflow(f"metrics_workflow_{i}", tasks)
                results = sdlc_manager.execute_workflow(workflow_id)
                
                # Record in monitoring system
                monitor.metrics_collector.record_workflow_execution(
                    workflow_id, f"Metrics Workflow {i}", results, "test"
                )
            
            # Generate comprehensive report
            report_path = project_path / "sdlc_report.json"
            monitor.generate_report(str(report_path))
            
            assert report_path.exists()
            
            # Verify report contents
            with open(report_path, 'r') as f:
                report_data = f.read()
                assert "metrics" in report_data
                assert "vulnerability_summary" in report_data
                
        finally:
            monitor.cleanup()


class TestSecurityIntegration:
    """Test security integration in realistic development workflows."""
    
    def test_security_in_ci_pipeline(self, ml_project_structure):
        """Test integration of security scanning in CI pipeline."""
        project_path = ml_project_structure
        
        # Create automated workflow with security scanning
        automated_workflow = AutomatedWorkflow(str(project_path))
        security_validator = SecurityValidator(str(project_path), SecurityLevel.STANDARD)
        
        try:
            automated_workflow.start()
            
            # Create CI workflow that includes security scanning
            ci_tasks = [
                WorkflowTask(
                    id="code_checkout",
                    name="Code Checkout",
                    command="echo 'Code checked out'",
                    description="Checkout code from repository"
                ),
                WorkflowTask(
                    id="dependency_install",
                    name="Install Dependencies",
                    command="echo 'Dependencies installed'",
                    description="Install project dependencies",
                    dependencies=["code_checkout"]
                ),
                WorkflowTask(
                    id="security_scan",
                    name="Security Scan",
                    command="echo 'Security scanning...' && sleep 1 && echo 'Security scan complete'",
                    description="Run security vulnerability scan",
                    dependencies=["dependency_install"]
                ),
                WorkflowTask(
                    id="unit_tests",
                    name="Unit Tests",
                    command="echo 'Running tests...' && sleep 1 && echo 'Tests passed'",
                    description="Run unit test suite",
                    dependencies=["security_scan"]
                )
            ]
            
            # Execute secure CI workflow
            workflow_id = automated_workflow.create_workflow("secure_ci", ci_tasks)
            execution_id = automated_workflow.execute_workflow(workflow_id)
            
            # Wait for execution
            time.sleep(4)
            
            # Run actual security scan
            scan_result = security_validator.run_comprehensive_scan()
            
            # Verify security scan found the planted vulnerabilities
            assert len(scan_result.vulnerabilities) >= 1  # Should find secrets in config.py
            
            # Test security policy validation
            is_valid, message = security_validator.validate_scan_result(scan_result)
            
            # Generate security metrics
            security_metrics = security_validator.get_security_metrics()
            assert "latest_scan" in security_metrics
            
        finally:
            automated_workflow.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])