"""Unit tests for security modules."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from continual_transformer.security.validator import (
    InputValidator, ModelValidator, SecurityValidator
)
from continual_transformer.security.sanitizer import (
    DataSanitizer, InputSanitizer, AdvancedSanitizer
)


class TestInputValidator:
    """Test suite for InputValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create test validator."""
        config = {
            "max_sequence_length": 100,
            "max_batch_size": 10
        }
        return InputValidator(config)
    
    def test_validate_text_input_valid(self, validator):
        """Test validation of valid text input."""
        text = "This is a normal text input."
        is_valid, warnings, sanitized = validator.validate_text_input(text)
        
        assert is_valid
        assert len(warnings) == 0
        assert sanitized == text
    
    def test_validate_text_input_list(self, validator):
        """Test validation of text list."""
        texts = ["Text one", "Text two", "Text three"]
        is_valid, warnings, sanitized = validator.validate_text_input(texts)
        
        assert is_valid
        assert len(warnings) == 0
        assert sanitized == texts
    
    def test_validate_text_input_empty(self, validator):
        """Test validation of empty text."""
        text = ""
        is_valid, warnings, sanitized = validator.validate_text_input(text)
        
        assert is_valid  # Empty is allowed
        assert "Empty or whitespace-only input" in " ".join(warnings)
        assert sanitized == text
    
    def test_validate_text_input_too_long(self, validator):
        """Test validation of overly long text."""
        long_text = "x" * 2000  # Exceeds max_sequence_length * 10
        is_valid, warnings, sanitized = validator.validate_text_input(long_text)
        
        assert is_valid  # Should be valid but truncated
        assert any("too long" in warning for warning in warnings)
        assert len(sanitized) < len(long_text)
    
    def test_validate_text_input_suspicious_content(self, validator):
        """Test validation of suspicious content."""
        suspicious_text = "<script>alert('xss')</script>Normal text"
        is_valid, warnings, sanitized = validator.validate_text_input(suspicious_text)
        
        assert is_valid  # Should be valid after sanitization
        assert any("Suspicious pattern" in warning for warning in warnings)
        assert "script" not in sanitized.lower()
    
    def test_validate_text_input_strict_mode(self, validator):
        """Test strict mode validation."""
        suspicious_text = "<script>alert('test')</script>"
        is_valid, warnings, sanitized = validator.validate_text_input(
            suspicious_text, strict_mode=True
        )
        
        assert not is_valid  # Should be invalid in strict mode
        assert any("Security violations" in warning for warning in warnings)
    
    def test_validate_tensor_input_valid(self, validator):
        """Test validation of valid tensor."""
        tensor = torch.randn(2, 10)
        is_valid, warnings = validator.validate_tensor_input(
            tensor, expected_shape=(2, 10), expected_dtype=torch.float32
        )
        
        assert is_valid
        assert len(warnings) == 0
    
    def test_validate_tensor_input_wrong_shape(self, validator):
        """Test validation of wrong tensor shape."""
        tensor = torch.randn(3, 5)
        is_valid, warnings = validator.validate_tensor_input(
            tensor, expected_shape=(2, 10)
        )
        
        assert not is_valid
        assert any("Shape mismatch" in warning for warning in warnings)
    
    def test_validate_tensor_input_nan_values(self, validator):
        """Test validation of tensor with NaN values."""
        tensor = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]])
        is_valid, warnings = validator.validate_tensor_input(tensor)
        
        assert not is_valid
        assert any("NaN values" in warning for warning in warnings)
    
    def test_validate_tensor_input_inf_values(self, validator):
        """Test validation of tensor with infinite values."""
        tensor = torch.tensor([[1.0, 2.0], [float('inf'), 4.0]])
        is_valid, warnings = validator.validate_tensor_input(tensor)
        
        assert not is_valid
        assert any("infinite values" in warning for warning in warnings)
    
    def test_validate_labels_valid(self, validator):
        """Test validation of valid labels."""
        labels = torch.tensor([0, 1, 2, 1, 0])
        is_valid, warnings = validator.validate_labels(labels, num_classes=3, task_id="test")
        
        assert is_valid
        assert len(warnings) == 0
    
    def test_validate_labels_negative(self, validator):
        """Test validation of negative labels."""
        labels = torch.tensor([0, -1, 2])
        is_valid, warnings = validator.validate_labels(labels, num_classes=3, task_id="test")
        
        assert not is_valid
        assert any("negative values" in warning for warning in warnings)
    
    def test_validate_labels_out_of_range(self, validator):
        """Test validation of labels exceeding class range."""
        labels = torch.tensor([0, 1, 5])  # 5 exceeds num_classes=3
        is_valid, warnings = validator.validate_labels(labels, num_classes=3, task_id="test")
        
        assert not is_valid
        assert any("exceed class range" in warning for warning in warnings)
    
    def test_validate_labels_single_class(self, validator):
        """Test validation of labels with single class."""
        labels = torch.tensor([1, 1, 1, 1])
        is_valid, warnings = validator.validate_labels(labels, num_classes=3, task_id="test")
        
        assert is_valid  # Valid but warning
        assert any("same class" in warning for warning in warnings)


class TestModelValidator:
    """Test suite for ModelValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create test validator."""
        return ModelValidator()
    
    def test_validate_model_state_valid(self, validator):
        """Test validation of valid model state."""
        state_dict = {
            "model_state_dict": {
                "layer1.weight": torch.randn(10, 5),
                "layer1.bias": torch.randn(10)
            },
            "config": {"model_name": "test"},
            "task_performance": {}
        }
        
        is_valid, warnings = validator.validate_model_state(state_dict)
        assert is_valid
        assert len(warnings) == 0
    
    def test_validate_model_state_missing_keys(self, validator):
        """Test validation of model state with missing keys."""
        state_dict = {
            "model_state_dict": {}
            # Missing "config" and "task_performance"
        }
        
        is_valid, warnings = validator.validate_model_state(state_dict)
        assert not is_valid
        assert any("Missing required keys" in warning for warning in warnings)
    
    def test_validate_model_state_nan_parameters(self, validator):
        """Test validation of model state with NaN parameters."""
        state_dict = {
            "model_state_dict": {
                "layer1.weight": torch.tensor([[1.0, float('nan')], [2.0, 3.0]])
            },
            "config": {},
            "task_performance": {}
        }
        
        is_valid, warnings = validator.validate_model_state(state_dict)
        assert not is_valid
        assert any("NaN values" in warning for warning in warnings)
    
    def test_validate_config_valid(self, validator):
        """Test validation of valid config."""
        config = {
            "model_name": "test-model",
            "max_tasks": 10,
            "device": "cpu",
            "learning_rate": 0.001
        }
        
        is_valid, warnings = validator.validate_config(config)
        assert is_valid
        assert len(warnings) == 0
    
    def test_validate_config_invalid_max_tasks(self, validator):
        """Test validation of config with invalid max_tasks."""
        config = {
            "model_name": "test",
            "max_tasks": -5,  # Invalid
            "device": "cpu"
        }
        
        is_valid, warnings = validator.validate_config(config)
        assert not is_valid
        assert any("Invalid max_tasks" in warning for warning in warnings)
    
    def test_validate_config_invalid_learning_rate(self, validator):
        """Test validation of config with invalid learning rate."""
        config = {
            "model_name": "test",
            "max_tasks": 10,
            "device": "cpu",
            "learning_rate": -0.1  # Invalid
        }
        
        is_valid, warnings = validator.validate_config(config)
        assert not is_valid
        assert any("Invalid learning_rate" in warning for warning in warnings)


class TestSecurityValidator:
    """Test suite for SecurityValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create test validator."""
        return SecurityValidator()
    
    def test_validate_inference_request_valid(self, validator):
        """Test validation of valid inference request."""
        text = "This is a normal text input for classification."
        task_id = "sentiment_analysis"
        
        is_valid, report = validator.validate_inference_request(text, task_id)
        
        assert is_valid
        assert report["overall_status"] == "approved"
        assert report["input_validation"]["status"] == "valid"
        assert len(report["errors"]) == 0
    
    def test_validate_inference_request_suspicious(self, validator):
        """Test validation of suspicious request."""
        text = "<script>alert('xss')</script>DROP TABLE users;"
        task_id = "test_task"
        
        is_valid, report = validator.validate_inference_request(text, task_id)
        
        assert report["security_checks"]["risk_level"] == "high"
        assert len(report["warnings"]) > 0
    
    def test_validate_inference_request_invalid_task_id(self, validator):
        """Test validation with invalid task ID."""
        text = "Normal text"
        task_id = ""  # Empty task ID
        
        is_valid, report = validator.validate_inference_request(text, task_id)
        
        assert not is_valid
        assert any("Invalid task_id" in error for error in report["errors"])
    
    def test_validate_inference_request_with_model_state(self, validator):
        """Test validation with model state."""
        text = "Test input"
        task_id = "test"
        model_state = {
            "model_state_dict": {"param": torch.randn(2, 2)},
            "config": {"model_name": "test"},
            "task_performance": {}
        }
        
        is_valid, report = validator.validate_inference_request(
            text, task_id, model_state
        )
        
        assert "model_validation" in report
        assert report["model_validation"]["status"] == "valid"
    
    def test_security_event_logging(self, validator):
        """Test security event logging."""
        # Make some validation requests
        validator.validate_inference_request("normal text", "task1")
        validator.validate_inference_request("<script>alert('xss')</script>", "task2")
        validator.validate_inference_request("", "")  # Invalid request
        
        summary = validator.get_security_summary()
        
        assert summary["status"] == "active"
        assert summary["total_events"] == 3
        assert summary["rejected_requests"] >= 1  # At least the invalid one
        assert len(summary["recent_events"]) <= 10


class TestDataSanitizer:
    """Test suite for DataSanitizer."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create test sanitizer."""
        return DataSanitizer()
    
    def test_sanitize_text_normal(self, sanitizer):
        """Test sanitization of normal text."""
        text = "This is a normal sentence."
        result = sanitizer.sanitize_text(text)
        
        assert result == text
    
    def test_sanitize_text_html(self, sanitizer):
        """Test sanitization of HTML content."""
        text = "<div>Some content</div> with <b>bold</b> text"
        result = sanitizer.sanitize_text(text)
        
        assert "<div>" not in result
        assert "<b>" not in result
        assert "Some content" in result
        assert "bold" in result
    
    def test_sanitize_text_script(self, sanitizer):
        """Test sanitization of script tags."""
        text = "<script>alert('dangerous')</script>Safe content"
        result = sanitizer.sanitize_text(text)
        
        assert "<script>" not in result
        assert "alert" not in result
        assert "Safe content" in result
    
    def test_sanitize_text_repeated_chars(self, sanitizer):
        """Test sanitization of repeated characters."""
        text = "Wooooooooow this is amaaaaaazing!!!!!!"
        result = sanitizer.sanitize_text(text)
        
        # Should limit repeated characters to max 3
        assert "ooooo" not in result  # More than 3 o's should be reduced
        assert "Wooo" in result  # Up to 3 should remain
    
    def test_sanitize_text_excessive_whitespace(self, sanitizer):
        """Test sanitization of excessive whitespace."""
        text = "Text   with     too    much     whitespace"
        result = sanitizer.sanitize_text(text)
        
        assert "   " not in result  # Multiple spaces should be reduced
        assert "with too much whitespace" in result
    
    def test_sanitize_text_strict_mode(self, sanitizer):
        """Test strict mode sanitization."""
        text = "Text with $pecial ch@racters!"
        result = sanitizer.sanitize_text(text, strict_mode=True)
        
        # In strict mode, special characters should be removed
        assert "$" not in result
        assert "@" not in result
        assert "!" in result  # ! should remain as it's in allowed chars
    
    def test_sanitize_text_list(self, sanitizer):
        """Test sanitization of text list."""
        texts = [
            "Normal text",
            "<script>alert('xss')</script>",
            "Text    with   whitespace"
        ]
        
        results = sanitizer.sanitize_text(texts)
        
        assert len(results) == 3
        assert results[0] == "Normal text"
        assert "<script>" not in results[1]
        assert "   " not in results[2]
    
    def test_sanitize_batch(self, sanitizer):
        """Test batch sanitization."""
        batch = {
            "text": ["Normal text", "<div>HTML content</div>"],
            "labels": [0, 1],
            "other_field": "unchanged"
        }
        
        result = sanitizer.sanitize_batch(batch, text_keys=["text"])
        
        assert result["labels"] == [0, 1]  # Unchanged
        assert result["other_field"] == "unchanged"  # Unchanged
        assert "<div>" not in result["text"][1]  # Sanitized
        assert "HTML content" in result["text"][1]


class TestInputSanitizer:
    """Test suite for InputSanitizer."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create test sanitizer."""
        return InputSanitizer()
    
    def test_sanitize_tensor_normal(self, sanitizer):
        """Test sanitization of normal tensor."""
        tensor = torch.randn(2, 3)
        result = sanitizer.sanitize_tensor(tensor)
        
        assert torch.equal(result, tensor)  # Should be unchanged
    
    def test_sanitize_tensor_nan(self, sanitizer):
        """Test sanitization of tensor with NaN."""
        tensor = torch.tensor([[1.0, float('nan')], [2.0, 3.0]])
        result = sanitizer.sanitize_tensor(tensor, replace_nan=True)
        
        assert not torch.isnan(result).any()
        assert result[0, 0] == 1.0  # Original values preserved
        assert result[0, 1] == 0.0  # NaN replaced with 0
    
    def test_sanitize_tensor_inf(self, sanitizer):
        """Test sanitization of tensor with infinity."""
        tensor = torch.tensor([[1.0, float('inf')], [-float('inf'), 3.0]])
        result = sanitizer.sanitize_tensor(tensor, replace_inf=True)
        
        assert not torch.isinf(result).any()
        assert result[0, 0] == 1.0  # Original value preserved
        assert result[1, 1] == 3.0  # Original value preserved
        # Infinity values should be replaced with large finite values
        assert torch.isfinite(result).all()
    
    def test_sanitize_tensor_clipping(self, sanitizer):
        """Test tensor clipping."""
        tensor = torch.tensor([[-5.0, 0.0], [10.0, 2.0]])
        result = sanitizer.sanitize_tensor(tensor, clip_range=(-2.0, 5.0))
        
        expected = torch.tensor([[-2.0, 0.0], [5.0, 2.0]])
        assert torch.allclose(result, expected)
    
    def test_sanitize_model_input(self, sanitizer):
        """Test complete model input sanitization."""
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)  # Wrong dtype
        attention_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        labels = torch.tensor([0, -1], dtype=torch.float)  # Wrong dtype and negative
        
        sanitized_ids, sanitized_mask, sanitized_labels = sanitizer.sanitize_model_input(
            input_ids, attention_mask, labels, vocab_size=1000
        )
        
        # Check dtypes were corrected
        assert sanitized_ids.dtype == torch.long
        assert sanitized_mask.dtype == torch.long
        assert sanitized_labels.dtype == torch.long
        
        # Check negative label was clipped
        assert (sanitized_labels >= 0).all()
        
        # Check attention mask is binary
        assert torch.all((sanitized_mask == 0) | (sanitized_mask == 1))
    
    def test_sanitize_prediction_input(self, sanitizer):
        """Test prediction input sanitization."""
        text = "<script>alert('test')</script>Normal text"
        task_id = "task@with#special$chars"
        
        sanitized_text, sanitized_task_id, warnings = sanitizer.sanitize_prediction_input(
            text, task_id
        )
        
        # Text should be sanitized
        assert "<script>" not in sanitized_text
        assert "Normal text" in sanitized_text
        
        # Task ID should be sanitized
        assert "@" not in sanitized_task_id
        assert "#" not in sanitized_task_id
        assert "$" not in sanitized_task_id
        assert "task" in sanitized_task_id
        
        # Should have warnings about changes
        assert len(warnings) > 0


class TestAdvancedSanitizer:
    """Test suite for AdvancedSanitizer."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create test sanitizer."""
        return AdvancedSanitizer()
    
    def test_analyze_content_safety_safe(self, sanitizer):
        """Test safety analysis of safe content."""
        text = "This is a normal, safe text for analysis."
        analysis = sanitizer.analyze_content_safety(text)
        
        assert analysis["safe"]
        assert analysis["severity"] == "low"
        assert len(analysis["issues"]) == 0
    
    def test_analyze_content_safety_email(self, sanitizer):
        """Test safety analysis with email address."""
        text = "Contact me at user@example.com for more info."
        analysis = sanitizer.analyze_content_safety(text)
        
        assert "Email address detected" in analysis["issues"]
        assert analysis["severity"] in ["low", "medium"]
    
    def test_analyze_content_safety_repetitive(self, sanitizer):
        """Test safety analysis of repetitive content."""
        text = "spam spam spam spam spam spam spam spam spam spam"
        analysis = sanitizer.analyze_content_safety(text)
        
        assert "repetition detected" in " ".join(analysis["issues"]).lower()
        assert analysis["severity"] in ["medium", "high"]
    
    def test_comprehensive_sanitization(self, sanitizer):
        """Test comprehensive sanitization."""
        data = {
            "text": "<script>alert('xss')</script>Normal content",
            "content": ["Safe text", "user@email.com in text"],
            "labels": [0, 1],
            "other": "unchanged"
        }
        
        sanitized_data, report = sanitizer.comprehensive_sanitization(data)
        
        # Check sanitization was applied
        assert "<script>" not in sanitized_data["text"]
        assert "Normal content" in sanitized_data["text"]
        
        # Check other fields unchanged
        assert sanitized_data["labels"] == [0, 1]
        assert sanitized_data["other"] == "unchanged"
        
        # Check report
        assert "safety_analyses" in report
        assert "sanitization_actions" in report
        assert report["overall_safety"] in ["safe", "caution", "unsafe"]
    
    def test_content_cache(self, sanitizer):
        """Test content analysis caching."""
        text = "Test content for caching"
        
        # First analysis
        analysis1 = sanitizer.analyze_content_safety(text)
        
        # Second analysis (should use cache)
        analysis2 = sanitizer.analyze_content_safety(text)
        
        # Should be identical (from cache)
        assert analysis1 == analysis2
        
        # Cache should have the entry
        text_hash = hash(text)
        assert text_hash in sanitizer._content_cache