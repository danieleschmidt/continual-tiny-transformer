"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "model_size": "base",
        "max_tasks": 5,
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 6,
        "vocab_size": 50000,
        "max_seq_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
    }


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return [
        "This is a positive example.",
        "This is a negative example.", 
        "Another positive case here.",
        "Yet another negative case.",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return [1, 0, 1, 0]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    from unittest.mock import Mock
    
    tokenizer = Mock()
    tokenizer.encode.return_value = [101, 2023, 2003, 102]  # Mock token IDs
    tokenizer.decode.return_value = "This is a test"
    tokenizer.vocab_size = 50000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 102
    tokenizer.bos_token_id = 101
    
    return tokenizer


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_torch_state():
    """Reset PyTorch state between tests."""
    yield
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-add markers based on test location."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            
        # Auto-mark benchmark tests
        if "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# Skip GPU tests if no GPU available
def pytest_runtest_setup(item):
    """Skip GPU tests if CUDA is not available."""
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("GPU not available")