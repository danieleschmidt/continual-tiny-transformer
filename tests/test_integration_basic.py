"""
Basic integration tests for continual learning framework.
Tests core functionality without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path


class MockTorch:
    """Mock PyTorch for testing without dependencies."""
    
    class Tensor:
        def __init__(self, data=None, device="cpu", dtype=None):
            self.data = data or [[1, 2, 3]]
            self.device = device
            self.dtype = dtype
            self.shape = (1, 3) if data is None else (len(data), len(data[0]) if data else 0)
        
        def to(self, device):
            return MockTorch.Tensor(self.data, device, self.dtype)
        
        def size(self, dim=None):
            if dim is not None:
                return self.shape[dim]
            return self.shape
        
        def mean(self, dim=None):
            return MockTorch.Tensor([[2.0]])
        
        def argmax(self, dim=-1):
            return MockTorch.Tensor([1])
        
        def sum(self):
            return MockTorch.Tensor([6])
        
        def item(self):
            return 1.0
        
        def cpu(self):
            return self
        
        def numpy(self):
            return [[1, 2, 3]]
        
        def backward(self):
            pass
        
        def __getitem__(self, key):
            return MockTorch.Tensor([self.data[0]])
    
    class nn:
        class Module:
            def __init__(self):
                self.training = True
            
            def parameters(self):
                return [MockTorch.Tensor()]
            
            def named_parameters(self):
                return [("test_param", MockTorch.Tensor())]
            
            def state_dict(self):
                return {"test_param": MockTorch.Tensor()}
            
            def load_state_dict(self, state_dict):
                pass
            
            def train(self):
                self.training = True
            
            def eval(self):
                self.training = False
            
            def to(self, device):
                return self
            
            def forward(self, *args, **kwargs):
                return {"logits": MockTorch.Tensor(), "loss": MockTorch.Tensor()}
        
        class Linear(Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
        
        class ModuleDict(Module):
            def __init__(self, modules=None):
                super().__init__()
                self.modules_dict = modules or {}
            
            def __setitem__(self, key, module):
                self.modules_dict[key] = module
            
            def __getitem__(self, key):
                return self.modules_dict[key]
            
            def keys(self):
                return self.modules_dict.keys()
            
            def values(self):
                return self.modules_dict.values()
        
        class CrossEntropyLoss:
            def __call__(self, logits, labels):
                return MockTorch.Tensor([0.5])
        
        class Sequential(Module):
            def __init__(self, *layers):
                super().__init__()
                self.layers = layers
        
        class ReLU(Module):
            pass
        
        class Dropout(Module):
            def __init__(self, p=0.1):
                super().__init__()
                self.p = p
        
        class Embedding(Module):
            def __init__(self, num_embeddings, embedding_dim):
                super().__init__()
                self.weight = MockTorch.Tensor()
    
    class optim:
        class AdamW:
            def __init__(self, parameters, lr=1e-3, **kwargs):
                self.param_groups = [{"lr": lr}]
            
            def step(self):
                pass
            
            def zero_grad(self):
                pass
        
        class SGD:
            def __init__(self, parameters, lr=1e-3, **kwargs):
                self.param_groups = [{"lr": lr}]
            
            def step(self):
                pass
            
            def zero_grad(self):
                pass
        
        class lr_scheduler:
            class LinearLR:
                def __init__(self, optimizer, **kwargs):
                    self.optimizer = optimizer
                
                def step(self):
                    pass
            
            class CosineAnnealingLR:
                def __init__(self, optimizer, **kwargs):
                    self.optimizer = optimizer
                
                def step(self):
                    pass
    
    class cuda:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def empty_cache():
            pass
        
        @staticmethod
        def device_count():
            return 0
    
    @staticmethod
    def tensor(data, device="cpu", dtype=None):
        return MockTorch.Tensor(data, device, dtype)
    
    @staticmethod
    def zeros(*shape, device="cpu"):
        return MockTorch.Tensor([[0] * shape[-1]] * (shape[0] if len(shape) > 1 else 1), device)
    
    @staticmethod
    def ones(*shape, device="cpu"):
        return MockTorch.Tensor([[1] * shape[-1]] * (shape[0] if len(shape) > 1 else 1), device)
    
    @staticmethod
    def save(obj, path):
        pass
    
    @staticmethod
    def load(path, map_location=None):
        return {"model_state_dict": {}, "config": None}


class MockTransformers:
    """Mock transformers library."""
    
    class AutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            mock_model = MockTorch.nn.Module()
            mock_model.config = MockConfig()
            return mock_model
    
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return MockTokenizer()
    
    class AutoConfig:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return MockConfig()


class MockConfig:
    """Mock model configuration."""
    
    def __init__(self):
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.max_position_embeddings = 512


class MockTokenizer:
    """Mock tokenizer."""
    
    def __call__(self, text, **kwargs):
        return {
            "input_ids": MockTorch.Tensor([[1, 2, 3, 4]]),
            "attention_mask": MockTorch.Tensor([[1, 1, 1, 1]])
        }


# Mock external dependencies
sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = MockTorch.nn
sys.modules['torch.optim'] = MockTorch.optim
sys.modules['torch.cuda'] = MockTorch.cuda
sys.modules['transformers'] = MockTransformers()
sys.modules['numpy'] = Mock()
sys.modules['scipy'] = Mock()
sys.modules['sklearn'] = Mock()
sys.modules['datasets'] = Mock()
sys.modules['tokenizers'] = Mock()
sys.modules['tqdm'] = Mock()
sys.modules['wandb'] = Mock()
sys.modules['tensorboard'] = Mock()
sys.modules['hydra'] = Mock()
sys.modules['omegaconf'] = Mock()


class TestBasicIntegration(unittest.TestCase):
    """Test basic integration of continual learning components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Import after mocking
        from continual_transformer.core.config import ContinualConfig
        from continual_transformer.core.model import ContinualTransformer
        
        self.ContinualConfig = ContinualConfig
        self.ContinualTransformer = ContinualTransformer
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = self.ContinualConfig(
            model_name="bert-base-uncased",
            max_tasks=10,
            adaptation_method="activation"
        )
        
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.max_tasks, 10)
        self.assertEqual(config.adaptation_method, "activation")
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = self.ContinualConfig(
            model_name="bert-base-uncased",
            max_tasks=5
        )
        
        model = self.ContinualTransformer(config)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.config, config)
        self.assertIsNotNone(model.base_model)
        self.assertIsNotNone(model.task_router)
    
    def test_task_registration(self):
        """Test task registration."""
        config = self.ContinualConfig(model_name="bert-base-uncased")
        model = self.ContinualTransformer(config)
        
        # Register a task
        model.register_task("sentiment", num_labels=2)
        
        self.assertIn("sentiment", model.adapters)
        self.assertIn("sentiment", model.classification_heads)
        self.assertEqual(model.task_router.num_tasks, 1)
    
    def test_task_switching(self):
        """Test task switching functionality."""
        config = self.ContinualConfig(model_name="bert-base-uncased")
        model = self.ContinualTransformer(config)
        
        # Register tasks
        model.register_task("task1", num_labels=2)
        model.register_task("task2", num_labels=3)
        
        # Test task switching
        model.set_current_task("task1")
        self.assertEqual(model.current_task_id, "task1")
        
        model.set_current_task("task2")
        self.assertEqual(model.current_task_id, "task2")
    
    def test_forward_pass(self):
        """Test model forward pass."""
        config = self.ContinualConfig(model_name="bert-base-uncased")
        model = self.ContinualTransformer(config)
        
        # Register a task
        model.register_task("test_task", num_labels=2)
        model.set_current_task("test_task")
        
        # Create mock input
        input_ids = MockTorch.Tensor([[1, 2, 3, 4]])
        attention_mask = MockTorch.Tensor([[1, 1, 1, 1]])
        labels = MockTorch.Tensor([1])
        
        # Forward pass
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            task_id="test_task"
        )
        
        self.assertIn("logits", outputs)
        self.assertIn("loss", outputs)
        self.assertIn("hidden_states", outputs)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        config = self.ContinualConfig(model_name="bert-base-uncased")
        model = self.ContinualTransformer(config)
        
        # Register tasks
        model.register_task("task1", num_labels=2)
        model.register_task("task2", num_labels=3)
        
        # Get memory usage
        memory_stats = model.get_memory_usage()
        
        self.assertIn("total_parameters", memory_stats)
        self.assertIn("frozen_parameters", memory_stats)
        self.assertIn("trainable_parameters", memory_stats)
        self.assertIn("num_tasks", memory_stats)
        self.assertEqual(memory_stats["num_tasks"], 2)
    
    def test_prediction_interface(self):
        """Test prediction interface."""
        config = self.ContinualConfig(model_name="bert-base-uncased")
        model = self.ContinualTransformer(config)
        
        # Register a task
        model.register_task("sentiment", num_labels=2)
        
        # Test prediction
        result = model.predict("This is a test sentence", task_id="sentiment")
        
        self.assertIn("predictions", result)
        self.assertIn("probabilities", result)
        self.assertIn("task_id", result)
        self.assertEqual(result["task_id"], "sentiment")
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        config = self.ContinualConfig(model_name="bert-base-uncased")
        model = self.ContinualTransformer(config)
        
        # Register a task
        model.register_task("test_task", num_labels=2)
        
        # Save model
        save_path = os.path.join(self.temp_dir, "test_model")
        model.save_model(save_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(save_path, "model.pt")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "config.yaml")))
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        config = self.ContinualConfig(model_name="bert-base-uncased")
        model = self.ContinualTransformer(config)
        
        # Test invalid task access
        with self.assertRaises(ValueError):
            model.set_current_task("nonexistent_task")
        
        # Test forward pass without task
        input_ids = MockTorch.Tensor([[1, 2, 3]])
        
        with self.assertRaises(ValueError):
            model.forward(input_ids=input_ids)
    
    def test_resilience_components(self):
        """Test resilience components initialization."""
        from continual_transformer.resilience import CircuitBreaker, HealthMonitor
        
        # Test circuit breaker
        circuit_breaker = CircuitBreaker(failure_threshold=3)
        self.assertEqual(circuit_breaker.failure_threshold, 3)
        self.assertEqual(circuit_breaker.state, "CLOSED")
        
        # Test health monitor
        health_monitor = HealthMonitor(check_interval=1.0)
        self.assertEqual(health_monitor.check_interval, 1.0)
    
    def test_scaling_components(self):
        """Test scaling components initialization."""
        try:
            from continual_transformer.scaling import DistributedTrainingManager, AsyncInferenceEngine
            
            config = self.ContinualConfig(model_name="bert-base-uncased")
            model = self.ContinualTransformer(config)
            
            # Test distributed training manager
            dist_manager = DistributedTrainingManager(model)
            self.assertIsNotNone(dist_manager)
            
            # Test async inference engine
            async_engine = AsyncInferenceEngine(model, max_batch_size=16)
            self.assertEqual(async_engine.max_batch_size, 16)
            
        except ImportError:
            self.skipTest("Scaling components not available")
    
    def test_optimization_components(self):
        """Test optimization components."""
        try:
            from continual_transformer.optimization import PerformanceOptimizer
            
            config = self.ContinualConfig(model_name="bert-base-uncased")
            model = self.ContinualTransformer(config)
            
            # Test performance optimizer
            optimizer = PerformanceOptimizer(model, config)
            self.assertIsNotNone(optimizer)
            
        except ImportError:
            self.skipTest("Optimization components not available")


class TestConfigurationSystem(unittest.TestCase):
    """Test configuration system."""
    
    def setUp(self):
        """Set up test environment."""
        from continual_transformer.core.config import ContinualConfig
        self.ContinualConfig = ContinualConfig
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = self.ContinualConfig()
        
        # Check default values
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.max_tasks, 10)
        self.assertEqual(config.adaptation_method, "activation")
        self.assertTrue(config.freeze_base_model)
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = self.ContinualConfig(
            model_name="roberta-base",
            max_tasks=20,
            learning_rate=2e-5,
            batch_size=64
        )
        
        self.assertEqual(config.model_name, "roberta-base")
        self.assertEqual(config.max_tasks, 20)
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.batch_size, 64)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid values
        with self.assertRaises((ValueError, AssertionError)):
            self.ContinualConfig(max_tasks=0)
        
        with self.assertRaises((ValueError, AssertionError)):
            self.ContinualConfig(learning_rate=-1.0)


class TestTaskManagement(unittest.TestCase):
    """Test task management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        from continual_transformer.tasks import TaskManager, Task
        from continual_transformer.core.config import ContinualConfig
        
        self.TaskManager = TaskManager
        self.Task = Task
        self.config = ContinualConfig()
    
    def test_task_creation(self):
        """Test task creation."""
        task = self.Task(
            task_id="test_task",
            task_type="classification",
            num_labels=3
        )
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.task_type, "classification")
        self.assertEqual(task.num_labels, 3)
    
    def test_task_manager(self):
        """Test task manager functionality."""
        manager = self.TaskManager(self.config)
        
        # Add tasks
        manager.add_task("task1", "classification", {"num_labels": 2})
        manager.add_task("task2", "classification", {"num_labels": 3})
        
        # Check task count
        self.assertEqual(len(manager.tasks), 2)
        self.assertIn("task1", manager.tasks)
        self.assertIn("task2", manager.tasks)
        
        # Get task
        task1 = manager.get_task("task1")
        self.assertEqual(task1.task_id, "task1")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)