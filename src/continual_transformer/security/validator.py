"""Input validation and security checks for continual learning models."""

import re
import torch
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class InputValidator:
    """Comprehensive input validation for continual learning models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_sequence_length = self.config.get("max_sequence_length", 512)
        self.max_batch_size = self.config.get("max_batch_size", 1000)
        self.allowed_encodings = self.config.get("allowed_encodings", ["utf-8", "ascii"])
        
        # Compile regex patterns for efficiency
        self._suspicious_patterns = [
            re.compile(r'<script.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'data:text/html', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
        ]
        
        self._sql_injection_patterns = [
            re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b', re.IGNORECASE),
            re.compile(r';\s*(SELECT|INSERT|UPDATE|DELETE)', re.IGNORECASE),
            re.compile(r'\b(UNION|OR|AND)\s+\d+\s*=\s*\d+', re.IGNORECASE),
        ]
        
        self._path_traversal_patterns = [
            re.compile(r'\.\./', re.IGNORECASE),
            re.compile(r'\.\.\\\\', re.IGNORECASE),
            re.compile(r'%2e%2e%2f', re.IGNORECASE),
            re.compile(r'%2e%2e\\\\', re.IGNORECASE),
        ]
    
    def validate_text_input(
        self, 
        text: Union[str, List[str]], 
        strict_mode: bool = False
    ) -> Tuple[bool, List[str], Union[str, List[str]]]:
        """Validate and sanitize text input.
        
        Args:
            text: Input text or list of texts
            strict_mode: Whether to apply strict validation
            
        Returns:
            Tuple of (is_valid, warnings, sanitized_text)
        """
        is_list = isinstance(text, list)
        texts = text if is_list else [text]
        warnings = []
        sanitized_texts = []
        
        for i, txt in enumerate(texts):
            txt_warnings = []
            sanitized = txt
            
            # Basic type validation
            if not isinstance(txt, str):
                txt_warnings.append(f"Text {i}: Non-string input converted to string")
                sanitized = str(txt)
            
            # Encoding validation
            try:
                sanitized.encode('utf-8')
            except UnicodeEncodeError as e:
                txt_warnings.append(f"Text {i}: Encoding error - {str(e)}")
                sanitized = sanitized.encode('utf-8', errors='replace').decode('utf-8')
            
            # Length validation
            if len(sanitized) > self.max_sequence_length * 10:  # Allow some buffer
                txt_warnings.append(f"Text {i}: Input too long ({len(sanitized)} chars), truncating")
                sanitized = sanitized[:self.max_sequence_length * 10]
            
            # Security checks
            security_issues = self._check_security_patterns(sanitized)
            if security_issues:
                if strict_mode:
                    return False, [f"Text {i}: Security violations - {security_issues}"], text
                else:
                    txt_warnings.extend([f"Text {i}: {issue}" for issue in security_issues])
                    sanitized = self._sanitize_text(sanitized)
            
            # Content validation
            if not sanitized.strip():
                txt_warnings.append(f"Text {i}: Empty or whitespace-only input")
            
            sanitized_texts.append(sanitized)
            warnings.extend(txt_warnings)
        
        # Overall validation
        if len(texts) > self.max_batch_size:
            warnings.append(f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}")
            sanitized_texts = sanitized_texts[:self.max_batch_size]
        
        # Return in same format as input
        result = sanitized_texts if is_list else sanitized_texts[0]
        is_valid = len(warnings) == 0 or not strict_mode
        
        return is_valid, warnings, result
    
    def validate_tensor_input(
        self, 
        tensor: torch.Tensor, 
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None
    ) -> Tuple[bool, List[str]]:
        """Validate tensor inputs.
        
        Args:
            tensor: Input tensor
            expected_shape: Expected tensor shape (None dimensions are flexible)
            expected_dtype: Expected tensor dtype
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Basic validation
        if not isinstance(tensor, torch.Tensor):
            warnings.append("Input is not a torch.Tensor")
            return False, warnings
        
        # Shape validation
        if expected_shape is not None:
            if len(tensor.shape) != len(expected_shape):
                warnings.append(
                    f"Shape mismatch: expected {len(expected_shape)} dimensions, "
                    f"got {len(tensor.shape)}"
                )
                return False, warnings
            
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if expected is not None and actual != expected:
                    warnings.append(
                        f"Shape mismatch at dimension {i}: expected {expected}, got {actual}"
                    )
        
        # Data type validation
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            warnings.append(f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
        
        # Value validation
        if torch.isnan(tensor).any():
            warnings.append("Tensor contains NaN values")
            return False, warnings
        
        if torch.isinf(tensor).any():
            warnings.append("Tensor contains infinite values")
            return False, warnings
        
        # Memory validation
        tensor_size_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)
        max_tensor_size_mb = self.config.get("max_tensor_size_mb", 1000)
        
        if tensor_size_mb > max_tensor_size_mb:
            warnings.append(
                f"Tensor size {tensor_size_mb:.2f}MB exceeds maximum {max_tensor_size_mb}MB"
            )
        
        return len(warnings) == 0, warnings
    
    def validate_labels(
        self, 
        labels: torch.Tensor, 
        num_classes: int,
        task_id: str
    ) -> Tuple[bool, List[str]]:
        """Validate label tensors.
        
        Args:
            labels: Label tensor
            num_classes: Expected number of classes
            task_id: Task identifier for context
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Basic tensor validation
        is_valid, tensor_warnings = self.validate_tensor_input(
            labels, 
            expected_dtype=torch.long
        )
        warnings.extend(tensor_warnings)
        
        if not is_valid:
            return False, warnings
        
        # Label-specific validation
        if labels.dim() != 1:
            warnings.append(f"Labels must be 1D, got {labels.dim()}D")
            return False, warnings
        
        # Value range validation
        min_label = labels.min().item()
        max_label = labels.max().item()
        
        if min_label < 0:
            warnings.append(f"Task {task_id}: Labels contain negative values (min: {min_label})")
            return False, warnings
        
        if max_label >= num_classes:
            warnings.append(
                f"Task {task_id}: Labels exceed class range. "
                f"Max label: {max_label}, num_classes: {num_classes}"
            )
            return False, warnings
        
        # Class distribution check
        unique_labels = torch.unique(labels)
        if len(unique_labels) == 1:
            warnings.append(f"Task {task_id}: All labels are the same class ({unique_labels[0]})")
        elif len(unique_labels) < num_classes * 0.5:  # Less than 50% of classes present
            warnings.append(
                f"Task {task_id}: Only {len(unique_labels)}/{num_classes} classes present in batch"
            )
        
        return len([w for w in warnings if "exceed" in w or "negative" in w]) == 0, warnings
    
    def _check_security_patterns(self, text: str) -> List[str]:
        """Check text for security-related patterns."""
        issues = []
        
        # Check for suspicious patterns
        for pattern in self._suspicious_patterns:
            if pattern.search(text):
                issues.append(f"Suspicious pattern detected: {pattern.pattern[:50]}...")
        
        # Check for SQL injection
        for pattern in self._sql_injection_patterns:
            if pattern.search(text):
                issues.append("Potential SQL injection pattern detected")
                break
        
        # Check for path traversal
        for pattern in self._path_traversal_patterns:
            if pattern.search(text):
                issues.append("Path traversal pattern detected")
                break
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_char_ratio > 0.5:
            issues.append(f"High special character ratio: {special_char_ratio:.2f}")
        
        return issues
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing or replacing dangerous content."""
        sanitized = text
        
        # Remove suspicious patterns
        for pattern in self._suspicious_patterns:
            sanitized = pattern.sub('', sanitized)
        
        # Remove potential SQL injection
        for pattern in self._sql_injection_patterns:
            sanitized = pattern.sub('', sanitized)
        
        # Remove path traversal attempts
        for pattern in self._path_traversal_patterns:
            sanitized = pattern.sub('', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized

class ModelValidator:
    """Validator for model states and configurations."""
    
    def __init__(self):
        self.required_model_keys = [
            "model_state_dict", "config", "task_performance"
        ]
    
    def validate_model_state(
        self, 
        state_dict: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate model state dictionary."""
        warnings = []
        
        # Check required keys
        missing_keys = [key for key in self.required_model_keys if key not in state_dict]
        if missing_keys:
            warnings.append(f"Missing required keys: {missing_keys}")
            return False, warnings
        
        # Validate state dict
        model_state = state_dict.get("model_state_dict", {})
        if not isinstance(model_state, dict):
            warnings.append("Invalid model_state_dict format")
            return False, warnings
        
        # Check for suspicious parameter names or values
        for param_name, param_tensor in model_state.items():
            if not isinstance(param_name, str):
                warnings.append(f"Invalid parameter name type: {type(param_name)}")
                continue
            
            if isinstance(param_tensor, torch.Tensor):
                # Check for NaN or inf values
                if torch.isnan(param_tensor).any():
                    warnings.append(f"Parameter {param_name} contains NaN values")
                if torch.isinf(param_tensor).any():
                    warnings.append(f"Parameter {param_name} contains infinite values")
        
        return len(warnings) == 0, warnings
    
    def validate_config(
        self, 
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate model configuration."""
        warnings = []
        
        required_config_keys = [
            "model_name", "max_tasks", "device"
        ]
        
        missing_keys = [key for key in required_config_keys if key not in config]
        if missing_keys:
            warnings.append(f"Missing required config keys: {missing_keys}")
        
        # Validate specific config values
        if "max_tasks" in config:
            max_tasks = config["max_tasks"]
            if not isinstance(max_tasks, int) or max_tasks <= 0:
                warnings.append(f"Invalid max_tasks value: {max_tasks}")
            elif max_tasks > 1000:
                warnings.append(f"Suspiciously high max_tasks: {max_tasks}")
        
        if "learning_rate" in config:
            lr = config["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1.0:
                warnings.append(f"Invalid learning_rate: {lr}")
        
        return len(warnings) == 0, warnings

class SecurityValidator:
    """High-level security validation orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.input_validator = InputValidator(config)
        self.model_validator = ModelValidator()
        self.security_log = []
    
    def validate_inference_request(
        self,
        text: Union[str, List[str]],
        task_id: str,
        model_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive validation for inference requests.
        
        Args:
            text: Input text(s)
            task_id: Task identifier
            model_state: Optional model state for validation
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        validation_report = {
            "timestamp": "now",  # Simplified timestamp
            "input_validation": {},
            "security_checks": {},
            "model_validation": {},
            "overall_status": "unknown",
            "warnings": [],
            "errors": []
        }
        
        # Input validation
        try:
            is_valid, warnings, sanitized_text = self.input_validator.validate_text_input(
                text, strict_mode=False
            )
            
            validation_report["input_validation"] = {
                "status": "valid" if is_valid else "invalid",
                "warnings": warnings,
                "sanitized": sanitized_text != text
            }
            
            validation_report["warnings"].extend(warnings)
            
        except Exception as e:
            error_msg = f"Input validation failed: {str(e)}"
            validation_report["errors"].append(error_msg)
            validation_report["input_validation"]["status"] = "error"
            logger.error(error_msg)
        
        # Task ID validation
        try:
            if not isinstance(task_id, str) or not task_id.strip():
                validation_report["errors"].append("Invalid task_id")
            elif not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
                validation_report["warnings"].append("Task ID contains unusual characters")
                
        except Exception as e:
            validation_report["errors"].append(f"Task ID validation failed: {str(e)}")
        
        # Model validation (if provided)
        if model_state is not None:
            try:
                model_valid, model_warnings = self.model_validator.validate_model_state(model_state)
                validation_report["model_validation"] = {
                    "status": "valid" if model_valid else "invalid",
                    "warnings": model_warnings
                }
                validation_report["warnings"].extend(model_warnings)
                
            except Exception as e:
                error_msg = f"Model validation failed: {str(e)}"
                validation_report["errors"].append(error_msg)
                validation_report["model_validation"]["status"] = "error"
        
        # Security assessment
        validation_report["security_checks"] = self._assess_security_risk(text, task_id)
        
        # Overall status determination
        if validation_report["errors"]:
            validation_report["overall_status"] = "rejected"
            overall_valid = False
        elif len(validation_report["warnings"]) > 10:  # Too many warnings
            validation_report["overall_status"] = "suspicious"
            overall_valid = False
        else:
            validation_report["overall_status"] = "approved"
            overall_valid = True
        
        # Log security event
        self._log_security_event(validation_report)
        
        return overall_valid, validation_report
    
    def _assess_security_risk(self, text: Union[str, List[str]], task_id: str) -> Dict[str, Any]:
        """Assess security risk of the request."""
        texts = text if isinstance(text, list) else [text]
        
        risk_assessment = {
            "risk_level": "low",
            "indicators": [],
            "recommendations": []
        }
        
        total_length = sum(len(t) for t in texts)
        
        # Length-based risk
        if total_length > 10000:
            risk_assessment["indicators"].append("Very long input")
            risk_assessment["risk_level"] = "medium"
        
        # Pattern-based risk
        for i, txt in enumerate(texts):
            security_issues = self.input_validator._check_security_patterns(txt)
            if security_issues:
                risk_assessment["indicators"].extend([f"Text {i}: {issue}" for issue in security_issues])
                risk_assessment["risk_level"] = "high"
        
        # Frequency-based risk (simplified)
        if len(texts) > 100:
            risk_assessment["indicators"].append("Large batch request")
            if risk_assessment["risk_level"] == "low":
                risk_assessment["risk_level"] = "medium"
        
        # Generate recommendations
        if risk_assessment["risk_level"] == "high":
            risk_assessment["recommendations"].extend([
                "Consider rejecting request",
                "Apply strict input sanitization",
                "Enable additional monitoring"
            ])
        elif risk_assessment["risk_level"] == "medium":
            risk_assessment["recommendations"].extend([
                "Apply input sanitization",
                "Monitor request patterns"
            ])
        
        return risk_assessment
    
    def _log_security_event(self, validation_report: Dict[str, Any]):
        """Log security validation events."""
        event = {
            "timestamp": validation_report["timestamp"],
            "status": validation_report["overall_status"],
            "risk_level": validation_report["security_checks"].get("risk_level", "unknown"),
            "error_count": len(validation_report["errors"]),
            "warning_count": len(validation_report["warnings"])
        }
        
        self.security_log.append(event)
        
        # Keep only last 1000 events
        if len(self.security_log) > 1000:
            self.security_log = self.security_log[-1000:]
        
        # Log high-risk events
        if event["risk_level"] == "high" or event["status"] == "rejected":
            logger.warning(f"High-risk security event: {event}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security events."""
        if not self.security_log:
            return {"status": "no_data", "events": 0}
        
        total_events = len(self.security_log)
        rejected_count = sum(1 for e in self.security_log if e["status"] == "rejected")
        high_risk_count = sum(1 for e in self.security_log if e["risk_level"] == "high")
        
        return {
            "status": "active",
            "total_events": total_events,
            "rejected_requests": rejected_count,
            "high_risk_requests": high_risk_count,
            "rejection_rate": rejected_count / total_events if total_events > 0 else 0,
            "recent_events": self.security_log[-10:]  # Last 10 events
        }

__all__ = ["InputValidator", "ModelValidator", "SecurityValidator"]