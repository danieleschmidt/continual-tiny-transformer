"""Example unit test to verify test setup."""

import pytest


@pytest.mark.unit
def test_example():
    """Example test to verify pytest is working."""
    assert True


@pytest.mark.unit
def test_sample_config(sample_config):
    """Test that sample config fixture works."""
    assert sample_config["model_size"] == "base"
    assert sample_config["max_tasks"] == 5


@pytest.mark.unit 
def test_sample_data(sample_text_data, sample_labels):
    """Test that sample data fixtures work."""
    assert len(sample_text_data) == len(sample_labels)
    assert len(sample_text_data) == 4