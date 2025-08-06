"""Comprehensive unit tests for ContinualTransformer."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from continual_transformer import ContinualTransformer, ContinualConfig
from continual_transformer.tasks.manager import TaskType, TaskStatus, TaskManager
from continual_transformer.data.loaders import TaskDataset, create_synthetic_task_data
from continual_transformer.monitoring.health import RobustHealthMonitor


class TestContinualConfig:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ContinualConfig()
        
        assert config.model_name == "distilbert-base-uncased"
        assert config.max_tasks == 50
        assert config.freeze_base_model is True
        assert config.adaptation_method == "activation"
        assert 0 < config.learning_rate < 1
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = ContinualConfig(adaptation_method="activation", task_routing_method="learned")
        assert config.adaptation_method == "activation"
        
        # Invalid adaptation method should raise
        with pytest.raises(ValueError, match="adaptation_method must be one of"):
            ContinualConfig(adaptation_method="invalid_method")
        
        # Invalid routing method should raise
        with pytest.raises(ValueError, match="task_routing_method must be one of"):
            ContinualConfig(task_routing_method="invalid_routing")
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ContinualConfig(
            model_name="bert-base-uncased",
            max_tasks=25,
            learning_rate=1e-5
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["model_name"] == "bert-base-uncased"
        assert config_dict["max_tasks"] == 25
        
        # Test YAML serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            
            # Load back
            loaded_config = ContinualConfig.from_yaml(f.name)
            assert loaded_config.model_name == "bert-base-uncased"
            assert loaded_config.max_tasks == 25


class TestTaskManager:
    """Test task management system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ContinualConfig(max_tasks=5)
        self.task_manager = TaskManager(self.config)
    
    def test_task_creation(self):
        """Test task creation and registration."""
        task = self.task_manager.add_task(
            task_id="test_task",
            task_type="classification",
            num_labels=2,
            description="Test task"
        )
        
        assert task.task_id == "test_task"
        assert task.task_type == TaskType.CLASSIFICATION
        assert task.num_labels == 2
        assert task.status == TaskStatus.REGISTERED
        assert len(self.task_manager) == 1
    
    def test_task_limit_enforcement(self):
        """Test task limit enforcement."""
        # Add tasks up to limit
        for i in range(self.config.max_tasks):
            self.task_manager.add_task(
                task_id=f"task_{i}",
                task_type="classification",
                num_labels=2
            )
        
        # Adding one more should raise error
        with pytest.raises(ValueError, match="Maximum number of tasks"):
            self.task_manager.add_task(
                task_id="overflow_task",
                task_type="classification",
                num_labels=2
            )
    
    def test_task_dependencies(self):
        """Test task dependency management."""
        # Create prerequisite task
        self.task_manager.add_task("task1", "classification", 2)
        
        # Create dependent task
        self.task_manager.add_task("task2", "classification", 2, prerequisites=["task1"])
        
        # Check dependency is recorded
        assert "task2" in self.task_manager.task_dependencies
        assert "task1" in self.task_manager.task_dependencies["task2"]
        
        # Check learning order respects dependencies
        learning_order = self.task_manager.get_learning_order()
        task1_index = learning_order.index("task1")
        task2_index = learning_order.index("task2")
        assert task1_index < task2_index
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        self.task_manager.add_task("task1", "classification", 2)
        self.task_manager.add_task("task2", "classification", 2, prerequisites=["task1"])
        
        # Manually create circular dependency
        self.task_manager.task_dependencies["task1"] = {"task2"}
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            self.task_manager.get_learning_order()
    
    def test_task_performance_tracking(self):
        """Test task performance tracking."""
        task = self.task_manager.add_task("perf_task", "classification", 2)
        
        # Update performance metrics
        metrics = {"accuracy": 0.85, "loss": 0.25}
        self.task_manager.update_task_performance("perf_task", metrics, memory_usage_mb=100, parameter_count=1000)
        
        # Check metrics were recorded
        assert task.best_accuracy == 0.85
        assert task.best_loss == 0.25
        assert task.memory_usage_mb == 100
        assert task.parameter_count == 1000
        assert len(task.training_history) == 1
    
    def test_forgetting_computation(self):
        """Test forgetting metric computation."""
        self.task_manager.add_task("old_task", "classification", 2)
        self.task_manager.add_task("new_task", "classification", 2)
        
        # Compute forgetting
        self.task_manager.compute_forgetting("old_task", "new_task", 0.9, 0.8)
        
        # Check forgetting was recorded
        assert "old_task" in self.task_manager.forgetting_matrix
        assert "new_task" in self.task_manager.forgetting_matrix["old_task"]
        assert self.task_manager.forgetting_matrix["old_task"]["new_task"] == 0.1


class TestTaskDataset:
    """Test dataset handling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102, 103, 0, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
        }
        
        self.test_data = [
            {"text": "positive example", "label": 1},
            {"text": "negative example", "label": 0},
            {"text": "neutral example", "label": "neutral"}
        ]
    
    def test_dataset_creation(self):
        """Test dataset creation and processing."""
        label_mapping = {"neutral": 2}
        
        dataset = TaskDataset(
            data=self.test_data,
            task_id="test_task",
            tokenizer=self.mock_tokenizer,
            max_length=128,
            label_mapping=label_mapping
        )
        
        assert len(dataset) == 3
        assert dataset.task_id == "test_task"
        
        # Check first item processing
        item = dataset[0]
        assert item["task_id"] == "test_task"
        assert torch.is_tensor(item["input_ids"])
        assert torch.is_tensor(item["labels"])
    
    def test_label_mapping(self):
        """Test string label mapping."""
        label_mapping = {"neutral": 2}
        
        dataset = TaskDataset(
            data=self.test_data,
            task_id="test_task",
            tokenizer=self.mock_tokenizer,
            label_mapping=label_mapping
        )
        
        # Check that string label was mapped
        neutral_item = dataset[2]
        assert neutral_item["labels"].item() == 2
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        dataset = TaskDataset(
            data=self.test_data,
            task_id="test_task",
            tokenizer=self.mock_tokenizer
        )
        
        stats = dataset.get_statistics()
        assert stats["task_id"] == "test_task"
        assert stats["num_samples"] == 3
        assert "label_distribution" in stats
        assert "avg_text_length" in stats


class TestContinualTransformerCore:
    """Test core ContinualTransformer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ContinualConfig(
            model_name="distilbert-base-uncased",
            max_tasks=5,
            device="cpu",
            num_epochs=1
        )
    
    @patch('continual_transformer.core.model.AutoModel')
    @patch('continual_transformer.core.model.AutoTokenizer')
    def test_model_initialization(self, mock_tokenizer, mock_model):
        """Test model initialization."""
        # Mock the transformer model
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 768
        mock_base_model.config.num_hidden_layers = 6
        mock_base_model.parameters.return_value = [torch.tensor([1.0, 2.0])]
        mock_model.from_pretrained.return_value = mock_base_model
        
        model = ContinualTransformer(self.config)
        
        assert model.config == self.config
        assert model.base_model == mock_base_model
        assert model.current_task_id is None
    
    @patch('continual_transformer.core.model.AutoModel')
    def test_task_registration(self, mock_model):
        """Test task registration process."""
        # Setup mocks
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 768
        mock_base_model.config.num_hidden_layers = 6
        mock_base_model.parameters.return_value = []
        mock_model.from_pretrained.return_value = mock_base_model
        
        model = ContinualTransformer(self.config)
        
        # Register a task
        model.register_task("sentiment", num_labels=2, task_type="classification")
        
        # Check task was registered
        assert "sentiment" in model.adapters
        assert "sentiment" in model.classification_heads
        assert model.task_router.num_tasks == 1
        assert "sentiment" in model.task_manager.tasks
    
    @patch('continual_transformer.core.model.AutoModel')
    def test_memory_usage_tracking(self, mock_model):
        """Test memory usage tracking."""
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 768
        mock_base_model.config.num_hidden_layers = 6
        
        # Mock parameters with specific counts
        base_params = [torch.randn(100), torch.randn(200)]  # 300 total
        mock_base_model.parameters.return_value = base_params
        mock_model.from_pretrained.return_value = mock_base_model
        
        model = ContinualTransformer(self.config)
        
        # Mock adapter parameters
        with patch.object(model, 'get_trainable_parameters') as mock_trainable:
            mock_trainable.return_value = [torch.randn(50)]  # 50 trainable
            
            memory_stats = model.get_memory_usage()
            
            assert "total_parameters" in memory_stats
            assert "trainable_parameters" in memory_stats
            assert "frozen_parameters" in memory_stats
            assert "num_tasks" in memory_stats


class TestHealthMonitoring:
    """Test health monitoring system."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        config = ContinualConfig()
        monitor = RobustHealthMonitor(config)
        
        assert monitor.config == config
        assert monitor.consecutive_failures == 0
        assert not monitor.health_check_disabled
    
    def test_error_recording(self):
        """Test error recording functionality."""
        monitor = RobustHealthMonitor()
        
        # Record some errors
        monitor._record_error("test_error", "Test error message")
        monitor._record_error("another_error", "Another error")
        monitor._record_error("test_error", "Second test error")
        
        error_report = monitor.get_error_report()
        
        assert error_report["total_errors"] == 3
        assert "test_error" in error_report["error_types"]
        assert "another_error" in error_report["error_types"]
        assert error_report["error_counts"]["test_error"] == 2
        assert error_report["error_counts"]["another_error"] == 1
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        monitor = RobustHealthMonitor()
        
        # Record task performance
        monitor.record_task_performance("task1", 10.5, success=True)
        monitor.record_task_performance("task2", 5.2, success=False)
        monitor.record_task_performance("task3", 8.7, success=True)
        
        perf_summary = monitor.get_performance_summary()
        
        assert perf_summary["total_tasks"] == 3
        assert perf_summary["successful_tasks"] == 2
        assert perf_summary["failed_tasks"] == 1
        assert perf_summary["success_rate_percent"] == pytest.approx(66.67, rel=1e-2)
    
    def test_health_check_circuit_breaker(self):
        """Test health check circuit breaker functionality."""
        monitor = RobustHealthMonitor()
        monitor.health_check_failure_threshold = 2
        
        # Mock health checker to always fail
        monitor.health_checker.comprehensive_health_check = Mock(side_effect=Exception("Test failure"))
        
        # First failure
        result1 = monitor.safe_health_check()
        assert result1["status"] == "ERROR"
        assert monitor.consecutive_failures == 1
        assert not monitor.health_check_disabled
        
        # Second failure - should disable health checks
        result2 = monitor.safe_health_check()
        assert result2["status"] == "ERROR"
        assert monitor.consecutive_failures == 2
        assert monitor.health_check_disabled
        
        # Third call should return disabled status
        result3 = monitor.safe_health_check()
        assert result3["status"] == "DISABLED"


class TestDataLoading:
    """Test data loading and processing."""
    
    def test_synthetic_data_creation(self):
        """Test synthetic data generation."""
        data = create_synthetic_task_data(
            task_id="test_task",
            num_samples=50,
            num_classes=3
        )
        
        assert len(data) == 50
        for item in data:
            assert "text" in item
            assert "label" in item
            assert "task_id" in item
            assert item["task_id"] == "test_task"
            assert 0 <= item["label"] < 3
    
    @patch('continual_transformer.data.loaders.AutoTokenizer')
    def test_dataloader_creation_from_file(self, mock_tokenizer):
        """Test dataloader creation from file."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[101, 102, 103]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create test data file
        test_data = [
            {"text": "test 1", "label": 0},
            {"text": "test 2", "label": 1}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            from continual_transformer.data.loaders import create_dataloader
            
            dataloader = create_dataloader(
                data_path=temp_path,
                batch_size=2,
                tokenizer_name="test-tokenizer"
            )
            
            assert dataloader is not None
            
        finally:
            Path(temp_path).unlink()  # Clean up


class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    @patch('continual_transformer.core.model.AutoModel')
    @patch('continual_transformer.core.model.AutoTokenizer')
    def test_sequential_task_learning_simulation(self, mock_tokenizer, mock_model):
        """Test sequential task learning simulation."""
        # Setup mocks
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 768
        mock_base_model.config.num_hidden_layers = 6
        mock_base_model.parameters.return_value = []
        mock_model.from_pretrained.return_value = mock_base_model
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        config = ContinualConfig(
            model_name="test-model",
            max_tasks=3,
            device="cpu"
        )
        
        model = ContinualTransformer(config)
        
        # Register multiple tasks
        tasks = [
            ("sentiment", 2, "classification"),
            ("topics", 5, "classification"),
            ("entities", 10, "sequence_labeling")
        ]
        
        for task_id, num_labels, task_type in tasks:
            model.register_task(task_id, num_labels, task_type)
        
        # Verify all tasks registered
        assert len(model.adapters) == 3
        assert model.task_router.num_tasks == 3
        
        # Check memory efficiency
        memory_stats = model.get_memory_usage()
        assert memory_stats["num_tasks"] == 3
    
    def test_error_recovery_scenarios(self):
        """Test error recovery and resilience."""
        monitor = RobustHealthMonitor()
        
        # Simulate various error scenarios
        error_scenarios = [
            ("connection_error", "Network timeout"),
            ("memory_error", "Out of GPU memory"),
            ("validation_error", "Invalid input data"),
            ("model_error", "Model inference failed")
        ]
        
        for error_type, error_msg in error_scenarios:
            monitor._record_error(error_type, error_msg)
        
        # Check error tracking
        error_report = monitor.get_error_report()
        assert error_report["total_errors"] == 4
        assert len(error_report["error_types"]) == 4
        
        # Test recovery
        monitor.reset_health_monitoring()
        assert not monitor.health_check_disabled
        assert monitor.consecutive_failures == 0


# Fixtures for pytest
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model_name": "distilbert-base-uncased",
        "max_tasks": 10,
        "hidden_size": 768,
        "learning_rate": 2e-5,
        "batch_size": 16
    }


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return [
        "This is a positive example",
        "This is a negative example",
        "Another positive case",
        "Another negative case"
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return [1, 0, 1, 0]


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        "--cov=continual_transformer",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-fail-under=85",
        "-v"
    ])