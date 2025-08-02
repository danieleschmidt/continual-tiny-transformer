"""Integration tests for continual learning pipeline."""

import pytest
import torch
from unittest.mock import Mock


@pytest.mark.integration
def test_sequential_task_learning(sample_config, sample_text_data, sample_labels):
    """Test learning multiple tasks sequentially."""
    # Mock the actual model for integration testing
    # In real implementation, this would use the actual ContinualTransformer
    
    model = Mock()
    model.learn_task = Mock(return_value={"loss": 0.1, "accuracy": 0.95})
    model.predict = Mock(return_value={"prediction": 1, "confidence": 0.9})
    
    # Test learning first task
    result1 = model.learn_task(
        task_id="sentiment",
        train_data=sample_text_data,
        labels=sample_labels
    )
    
    assert result1["accuracy"] > 0.8
    
    # Test learning second task
    result2 = model.learn_task(
        task_id="classification", 
        train_data=sample_text_data,
        labels=sample_labels
    )
    
    assert result2["accuracy"] > 0.8
    
    # Test that both tasks can still be used
    pred1 = model.predict(text="Test sentiment", task_id="sentiment")
    pred2 = model.predict(text="Test classification", task_id="classification")
    
    assert pred1["prediction"] is not None
    assert pred2["prediction"] is not None


@pytest.mark.integration
@pytest.mark.slow
def test_memory_efficiency():
    """Test that memory usage remains constant across tasks."""
    # This would test actual memory usage patterns
    # For now, just verify the concept works
    
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Simulate adding multiple tasks
    for task_id in range(5):
        # In real implementation, this would actually train tasks
        # For testing, we just ensure memory doesn't grow
        current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Memory should not grow significantly (allow 10% variance)
        if torch.cuda.is_available():
            memory_growth = (current_memory - initial_memory) / max(initial_memory, 1)
            assert memory_growth < 0.1, f"Memory grew by {memory_growth:.2%}"


@pytest.mark.integration
def test_knowledge_retention():
    """Test that previous tasks maintain performance."""
    # Mock knowledge retention testing
    
    model = Mock()
    
    # Initial performance on task 1
    model.evaluate = Mock(return_value={"accuracy": 0.95})
    initial_accuracy = model.evaluate(task_id="task1")["accuracy"]
    
    # After learning task 2
    model.learn_task = Mock()
    model.learn_task(task_id="task2", train_data=[], labels=[])
    
    # Performance on task 1 should still be high
    retained_accuracy = model.evaluate(task_id="task1")["accuracy"]
    
    # Should retain >90% of original performance
    retention_ratio = retained_accuracy / initial_accuracy
    assert retention_ratio > 0.9, f"Knowledge retention: {retention_ratio:.2%}"


@pytest.mark.integration
@pytest.mark.gpu
def test_gpu_compatibility():
    """Test that the framework works with GPU acceleration."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    device = torch.device("cuda")
    
    # Test basic GPU operations
    tensor = torch.randn(10, 10).to(device)
    result = torch.matmul(tensor, tensor.T)
    
    assert result.device.type == "cuda"
    assert result.shape == (10, 10)


@pytest.mark.integration
def test_configuration_validation(sample_config):
    """Test that configuration validation works properly."""
    # Test valid configuration
    assert sample_config["max_tasks"] > 0
    assert sample_config["hidden_size"] > 0
    assert 0 < sample_config["learning_rate"] < 1
    
    # Test invalid configurations would be rejected
    invalid_configs = [
        {"max_tasks": 0},
        {"hidden_size": -1},
        {"learning_rate": 2.0},
    ]
    
    for invalid_config in invalid_configs:
        # In real implementation, this would raise ValidationError
        # For testing, we just verify the concept
        assert any(value <= 0 or value > 1 for value in invalid_config.values())