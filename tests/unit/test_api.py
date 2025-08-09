"""Unit tests for the high-level API."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from continual_transformer.api import ContinualLearningAPI
from continual_transformer.core import ContinualConfig, ContinualTransformer


class TestContinualLearningAPI:
    """Test suite for ContinualLearningAPI."""
    
    @pytest.fixture
    def api(self):
        """Create test API instance."""
        with patch('continual_transformer.core.model.AutoModel'):
            with patch('continual_transformer.core.model.AutoTokenizer'):
                api = ContinualLearningAPI(
                    model_name="test-model",
                    max_tasks=5,
                    device="cpu"
                )
                return api
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        return [
            {"text": "This is positive", "label": 1},
            {"text": "This is negative", "label": 0},
            {"text": "Great product", "label": 1},
            {"text": "Terrible service", "label": 0},
        ]
    
    def test_initialization(self):
        """Test API initialization."""
        with patch('continual_transformer.core.model.AutoModel'):
            with patch('continual_transformer.core.model.AutoTokenizer'):
                api = ContinualLearningAPI(
                    model_name="test-model",
                    max_tasks=10
                )
                
                assert api.config.model_name == "test-model"
                assert api.config.max_tasks == 10
                assert isinstance(api.model, ContinualTransformer)
                assert api.trained_tasks == set()
    
    def test_add_task(self, api):
        """Test adding a new task."""
        api.add_task("sentiment", num_labels=2, task_type="classification")
        
        # Check task was registered
        assert "sentiment" in api.model.adapters
        assert "sentiment" in api.model.classification_heads
        assert api.model.classification_heads["sentiment"].out_features == 2
    
    def test_add_multiple_tasks(self, api):
        """Test adding multiple tasks."""
        api.add_task("sentiment", num_labels=2)
        api.add_task("topic", num_labels=3)
        api.add_task("emotion", num_labels=5)
        
        assert len(api.model.adapters) == 3
        assert "sentiment" in api.model.adapters
        assert "topic" in api.model.adapters
        assert "emotion" in api.model.adapters
    
    @patch('continual_transformer.data.loaders.create_dataloader')
    def test_train_task(self, mock_dataloader, api, sample_data):
        """Test training a task."""
        # Setup mocks
        mock_dataloader.return_value = Mock()
        api.model.learn_task = Mock()
        api.model.task_performance = {
            "sentiment": {
                "train_accuracy": [0.8],
                "eval_accuracy": [0.75],
                "train_loss": [0.5],
                "eval_loss": [0.6]
            }
        }
        
        # Add task and train
        api.add_task("sentiment", num_labels=2)
        metrics = api.train_task(
            task_id="sentiment",
            train_data=sample_data,
            epochs=3,
            batch_size=2
        )
        
        # Verify training was called
        api.model.learn_task.assert_called_once()
        assert "sentiment" in api.trained_tasks
        assert metrics["train_accuracy"] == 0.8
        assert metrics["train_loss"] == 0.5
    
    def test_predict_untrained_task(self, api):
        """Test prediction on untrained task shows warning."""
        api.add_task("sentiment", num_labels=2)
        
        # Mock model predict method
        api.model.predict = Mock(return_value={
            "predictions": [1],
            "probabilities": [[0.3, 0.7]]
        })
        
        with patch('builtins.print') as mock_print:
            result = api.predict("Test text", "sentiment")
            
            # Check warning was printed
            mock_print.assert_called_with("⚠️  Warning: Task 'sentiment' has not been trained yet")
            
            # Check prediction was still made
            api.model.predict.assert_called_once_with("Test text", "sentiment")
            assert result["predictions"] == [1]
    
    @patch('continual_transformer.data.loaders.create_dataloader')
    def test_evaluate_task(self, mock_dataloader, api, sample_data):
        """Test task evaluation."""
        mock_dataloader.return_value = Mock()
        api.model.evaluate_task = Mock(return_value={
            "loss": 0.4,
            "accuracy": 0.85,
            "total_samples": 100
        })
        
        api.add_task("sentiment", num_labels=2)
        metrics = api.evaluate_task("sentiment", sample_data)
        
        assert metrics["accuracy"] == 0.85
        assert metrics["loss"] == 0.4
        api.model.evaluate_task.assert_called_once()
    
    def test_get_memory_usage(self, api):
        """Test memory usage retrieval."""
        api.model.get_memory_usage = Mock(return_value={
            "total_parameters": 1000,
            "trainable_parameters": 100,
            "num_tasks": 2
        })
        
        usage = api.get_memory_usage()
        
        assert usage["total_parameters"] == 1000
        assert usage["trainable_parameters"] == 100
        assert usage["num_tasks"] == 2
    
    def test_get_task_info(self, api):
        """Test task information retrieval."""
        api.add_task("sentiment", num_labels=2)
        api.add_task("topic", num_labels=3)
        api.trained_tasks.add("sentiment")
        
        api.model.get_memory_usage = Mock(return_value={"total_parameters": 1000})
        
        info = api.get_task_info()
        
        assert "sentiment" in info["registered_tasks"]
        assert "topic" in info["registered_tasks"]
        assert "sentiment" in info["trained_tasks"]
        assert "topic" not in info["trained_tasks"]
        assert info["num_tasks"] == 2
        assert info["max_tasks"] == 5
    
    def test_save_and_load(self, api):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"
            
            # Add some tasks and mark as trained
            api.add_task("sentiment", num_labels=2)
            api.trained_tasks.add("sentiment")
            
            # Mock model save
            api.model.save_model = Mock()
            
            # Save
            api.save(str(save_path))
            api.model.save_model.assert_called_once_with(str(save_path))
            
            # Check metadata file was created
            metadata_file = save_path / "api_metadata.json"
            assert metadata_file.exists()
            
            # Test loading
            with patch('continual_transformer.core.model.ContinualTransformer.load_model') as mock_load:
                mock_model = Mock()
                mock_model.config = api.config
                mock_model.adapters = {"sentiment": Mock()}
                mock_load.return_value = mock_model
                
                loaded_api = ContinualLearningAPI.load(str(save_path))
                
                assert "sentiment" in loaded_api.trained_tasks
                mock_load.assert_called_once_with(str(save_path))
    
    def test_optimize_for_deployment(self, api):
        """Test deployment optimization."""
        api.model.optimize_for_inference = Mock(return_value={
            "torch_compile": True,
            "quantization": True
        })
        
        with patch('builtins.print') as mock_print:
            optimizations = api.optimize_for_deployment("speed")
            
            # Check optimization was called
            api.model.optimize_for_inference.assert_called_once_with("speed")
            
            # Check print statements
            assert mock_print.call_count >= 2  # At least start and completion messages
            
            assert "torch_compile" in optimizations
            assert "quantization" in optimizations
    
    def test_evaluate_all_tasks(self, api):
        """Test evaluation of all tasks."""
        # Setup tasks
        api.add_task("sentiment", num_labels=2)
        api.add_task("topic", num_labels=3)
        api.trained_tasks.add("sentiment")
        api.trained_tasks.add("topic")
        
        # Mock evaluate_task method
        def mock_evaluate(task_id, data):
            if task_id == "sentiment":
                return {"accuracy": 0.8, "loss": 0.3}
            else:
                return {"accuracy": 0.7, "loss": 0.4}
        
        api.evaluate_task = Mock(side_effect=mock_evaluate)
        
        # Mock metrics computation
        api.metrics.compute_continual_metrics = Mock(return_value={
            "average_accuracy": 0.75,
            "forgetting_rate": 0.1,
            "knowledge_retention": 0.9
        })
        
        eval_data = {
            "sentiment": [{"text": "test", "label": 1}],
            "topic": [{"text": "test", "label": 0}]
        }
        
        with patch('builtins.print') as mock_print:
            results = api.evaluate_all_tasks(eval_data)
            
            assert results["sentiment"]["accuracy"] == 0.8
            assert results["topic"]["accuracy"] == 0.7
            
            # Check metrics computation was called
            api.metrics.compute_continual_metrics.assert_called_once_with(results)
            
            # Check summary was printed
            assert any("Average Accuracy" in str(call) for call in mock_print.call_args_list)
    
    def test_evaluate_all_tasks_missing_data(self, api):
        """Test evaluation with missing data for some tasks."""
        api.add_task("sentiment", num_labels=2)
        api.add_task("topic", num_labels=3)
        api.trained_tasks.add("sentiment")
        api.trained_tasks.add("topic")
        
        api.evaluate_task = Mock(return_value={"accuracy": 0.8})
        
        eval_data = {
            "sentiment": [{"text": "test", "label": 1}]
            # Missing "topic" data
        }
        
        with patch('builtins.print') as mock_print:
            results = api.evaluate_all_tasks(eval_data)
            
            assert "sentiment" in results
            assert "topic" not in results
            
            # Check warning was printed for missing data
            assert any("No evaluation data provided for task 'topic'" in str(call) 
                     for call in mock_print.call_args_list)
    
    def test_error_handling_invalid_task(self, api):
        """Test error handling for invalid task operations."""
        # Test training non-existent task
        with pytest.raises(ValueError, match="not registered"):
            api.train_task("nonexistent", [])
        
        # Test prediction on non-existent task (should work but show warning)
        api.model.predict = Mock(side_effect=ValueError("Task not found"))
        
        with pytest.raises(ValueError):
            api.predict("test", "nonexistent")


class TestAPIIntegration:
    """Integration tests for the API."""
    
    def test_full_workflow_mock(self):
        """Test complete workflow with mocked components."""
        with patch('continual_transformer.core.model.AutoModel'):
            with patch('continual_transformer.core.model.AutoTokenizer'):
                with patch('continual_transformer.data.loaders.create_dataloader') as mock_loader:
                    # Create API
                    api = ContinualLearningAPI(model_name="test-model", device="cpu")
                    
                    # Mock dataloader
                    mock_loader.return_value = Mock()
                    
                    # Mock model methods
                    api.model.learn_task = Mock()
                    api.model.predict = Mock(return_value={
                        "predictions": [1],
                        "probabilities": [[0.2, 0.8]]
                    })
                    api.model.task_performance = {
                        "sentiment": {
                            "train_accuracy": [0.9],
                            "train_loss": [0.1],
                            "eval_accuracy": [],
                            "eval_loss": []
                        }
                    }
                    
                    # 1. Add task
                    api.add_task("sentiment", num_labels=2)
                    assert "sentiment" in api.model.adapters
                    
                    # 2. Train task
                    train_data = [{"text": "Great!", "label": 1}, {"text": "Bad", "label": 0}]
                    metrics = api.train_task("sentiment", train_data, epochs=1)
                    
                    assert "sentiment" in api.trained_tasks
                    assert metrics["train_accuracy"] == 0.9
                    
                    # 3. Make predictions
                    result = api.predict("This is awesome!", "sentiment")
                    assert result["predictions"] == [1]
                    assert len(result["probabilities"][0]) == 2
                    
                    # 4. Check task info
                    info = api.get_task_info()
                    assert info["num_tasks"] == 1
                    assert "sentiment" in info["trained_tasks"]