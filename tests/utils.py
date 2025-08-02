"""Testing utilities and helper functions."""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import tempfile
import shutil


def create_mock_dataset(
    num_samples: int = 100,
    seq_length: int = 128,
    vocab_size: int = 1000,
    num_classes: int = 2,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a mock dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        seq_length: Sequence length for each sample
        vocab_size: Vocabulary size for token generation
        num_classes: Number of classes for labels
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (input_ids, labels)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random token sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    return input_ids, labels


def create_mock_continual_tasks(
    num_tasks: int = 5,
    samples_per_task: int = 100,
    **kwargs
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Create multiple mock tasks for continual learning testing.
    
    Args:
        num_tasks: Number of tasks to create
        samples_per_task: Number of samples per task
        **kwargs: Additional arguments for create_mock_dataset
        
    Returns:
        Dictionary mapping task names to datasets
    """
    tasks = {}
    
    for i in range(num_tasks):
        task_name = f"task_{i}"
        input_ids, labels = create_mock_dataset(
            num_samples=samples_per_task,
            seed=42 + i,  # Different seed per task
            **kwargs
        )
        
        tasks[task_name] = {
            "input_ids": input_ids,
            "labels": labels,
            "task_id": i
        }
    
    return tasks


def assert_model_outputs_valid(outputs: Dict[str, torch.Tensor]) -> None:
    """Assert that model outputs are valid.
    
    Args:
        outputs: Dictionary of model outputs
    """
    assert isinstance(outputs, dict), "Outputs should be a dictionary"
    
    if "logits" in outputs:
        logits = outputs["logits"]
        assert isinstance(logits, torch.Tensor), "Logits should be a tensor"
        assert not torch.isnan(logits).any(), "Logits should not contain NaN"
        assert not torch.isinf(logits).any(), "Logits should not contain Inf"
    
    if "loss" in outputs:
        loss = outputs["loss"]
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"


def assert_memory_efficient(
    func,
    max_memory_growth_mb: float = 100.0,
    iterations: int = 5
) -> None:
    """Assert that a function is memory efficient.
    
    Args:
        func: Function to test
        max_memory_growth_mb: Maximum allowed memory growth in MB
        iterations: Number of iterations to test
    """
    if not torch.cuda.is_available():
        return  # Skip memory testing on CPU
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    for _ in range(iterations):
        func()
        torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated()
    memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
    
    assert memory_growth_mb < max_memory_growth_mb, (
        f"Memory grew by {memory_growth_mb:.1f}MB, "
        f"exceeds limit of {max_memory_growth_mb}MB"
    )


def create_temp_config(config_dict: Dict[str, Any]) -> Path:
    """Create a temporary configuration file.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Path to temporary config file
    """
    import yaml
    
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.yaml', 
        delete=False
    )
    
    yaml.dump(config_dict, temp_file, default_flow_style=False)
    temp_file.close()
    
    return Path(temp_file.name)


def cleanup_temp_files(file_paths: List[Path]) -> None:
    """Clean up temporary files.
    
    Args:
        file_paths: List of file paths to clean up
    """
    for file_path in file_paths:
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """Mock encoding function."""
        # Simple hash-based encoding for consistency
        tokens = [self.bos_token_id]
        for char in text[:max_length-2]:
            token_id = (hash(char) % (self.vocab_size - 3)) + 3
            tokens.append(token_id)
        tokens.append(self.eos_token_id)
        
        # Pad if necessary
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
            
        return tokens[:max_length]
    
    def decode(self, token_ids: List[int]) -> str:
        """Mock decoding function."""
        # Simple mock decoding
        return f"decoded_text_{len(token_ids)}_tokens"


def simulate_training_step(
    model,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer
) -> Dict[str, float]:
    """Simulate a training step for testing.
    
    Args:
        model: Model to train
        batch: Training batch
        optimizer: Optimizer
        
    Returns:
        Training metrics
    """
    model.train()
    optimizer.zero_grad()
    
    outputs = model(**batch)
    loss = outputs.get("loss", torch.tensor(0.0))
    
    loss.backward()
    optimizer.step()
    
    return {
        "loss": loss.item(),
        "gradient_norm": get_gradient_norm(model)
    }


def get_gradient_norm(model: torch.nn.Module) -> float:
    """Get the gradient norm of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)