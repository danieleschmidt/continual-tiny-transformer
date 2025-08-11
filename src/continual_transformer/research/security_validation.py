"""
Research-Grade Security and Validation Framework

This module implements comprehensive security measures and validation protocols for 
research-grade continual learning systems:
- Advanced input sanitization and validation
- Model integrity verification
- Secure multi-party computation protocols
- Differential privacy mechanisms
- Adversarial robustness testing
- Reproducibility validation
- Data provenance tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import re
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class SecurityValidationResult:
    """Result of security validation check."""
    passed: bool
    confidence_score: float
    risk_level: str  # low, medium, high, critical
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProvenanceRecord:
    """Record for tracking data provenance and lineage."""
    data_id: str
    source: str
    timestamp: float
    transformations: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: Optional[str] = None
    access_log: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedInputValidator:
    """Advanced input validation with ML-based anomaly detection."""
    
    def __init__(self, config):
        self.config = config
        
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Anomaly detection models
        self.input_anomaly_detector = IsolationForest(contamination=0.1)
        self.is_trained = False
        
        # Pattern-based filters
        self.malicious_patterns = self._load_malicious_patterns()
        
        # Input statistics for baseline establishment
        self.input_statistics = defaultdict(list)
        
    def _initialize_validation_rules(self) -> Dict[str, Callable]:
        """Initialize comprehensive validation rules."""
        
        return {
            'tensor_shape': self._validate_tensor_shape,
            'numerical_bounds': self._validate_numerical_bounds,
            'data_type': self._validate_data_type,
            'memory_constraints': self._validate_memory_constraints,
            'pattern_matching': self._validate_against_patterns,
            'statistical_bounds': self._validate_statistical_bounds,
            'injection_attacks': self._detect_injection_attacks,
            'adversarial_patterns': self._detect_adversarial_patterns
        }
    
    def _load_malicious_patterns(self) -> List[str]:
        """Load patterns for detecting malicious inputs."""
        
        return [
            r'<script[^>]*>.*?</script>',  # XSS patterns
            r'union\s+select',  # SQL injection
            r'../../../',  # Path traversal
            r'eval\s*\(',  # Code injection
            r'exec\s*\(',  # Code execution
            r'import\s+os',  # OS import
            r'__import__',  # Dynamic imports
        ]
    
    def validate_input(
        self, 
        input_data: Any, 
        validation_context: Dict[str, Any]
    ) -> SecurityValidationResult:
        """Comprehensive input validation with security checks."""
        
        violations = []
        recommendations = []
        confidence_scores = []
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(input_data, validation_context)
                
                if not result['passed']:
                    violations.extend(result.get('violations', []))
                    recommendations.extend(result.get('recommendations', []))
                
                confidence_scores.append(result.get('confidence', 0.5))
                
            except Exception as e:
                logger.error(f"Validation rule {rule_name} failed: {e}")
                violations.append(f"Validation rule {rule_name} execution failed")
        
        # Calculate overall confidence and risk level
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        risk_level = self._calculate_risk_level(violations, overall_confidence)
        
        return SecurityValidationResult(
            passed=len(violations) == 0,
            confidence_score=overall_confidence,
            risk_level=risk_level,
            violations=violations,
            recommendations=recommendations,
            metadata={
                'validation_timestamp': time.time(),
                'rules_executed': list(self.validation_rules.keys()),
                'input_hash': self._calculate_input_hash(input_data)
            }
        )
    
    def _validate_tensor_shape(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tensor shapes and dimensions."""
        
        if not isinstance(input_data, torch.Tensor):
            return {'passed': False, 'violations': ['Input is not a tensor']}
        
        violations = []
        
        # Check dimensional constraints
        max_dims = context.get('max_dimensions', 4)
        if input_data.dim() > max_dims:
            violations.append(f"Tensor has {input_data.dim()} dimensions, exceeding limit of {max_dims}")
        
        # Check shape constraints
        max_size = context.get('max_tensor_size', 1e8)
        if input_data.numel() > max_size:
            violations.append(f"Tensor has {input_data.numel()} elements, exceeding limit of {max_size}")
        
        # Check for suspicious shapes (potential adversarial inputs)
        if any(dim == 1337 or dim == 31337 for dim in input_data.shape):
            violations.append("Suspicious dimension sizes detected (potential adversarial input)")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'confidence': 0.9
        }
    
    def _validate_numerical_bounds(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numerical values are within acceptable bounds."""
        
        if not isinstance(input_data, torch.Tensor):
            return {'passed': True, 'confidence': 1.0}
        
        violations = []
        
        # Check for NaN or Inf values
        if torch.isnan(input_data).any():
            violations.append("Input contains NaN values")
        
        if torch.isinf(input_data).any():
            violations.append("Input contains infinite values")
        
        # Check value ranges
        if input_data.dtype in [torch.float32, torch.float64]:
            min_val, max_val = float(input_data.min()), float(input_data.max())
            
            # Detect suspiciously large values
            if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                violations.append(f"Extreme values detected: min={min_val}, max={max_val}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'confidence': 0.95
        }
    
    def _validate_data_type(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data types and formats."""
        
        violations = []
        expected_type = context.get('expected_type')
        
        if expected_type and not isinstance(input_data, expected_type):
            violations.append(f"Expected type {expected_type}, got {type(input_data)}")
        
        # Check for potentially dangerous types
        dangerous_types = [type(eval), type(exec), type(__import__)]
        if type(input_data) in dangerous_types:
            violations.append("Dangerous callable type detected")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'confidence': 0.8
        }
    
    def _validate_memory_constraints(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate memory usage constraints."""
        
        violations = []
        
        if isinstance(input_data, torch.Tensor):
            # Calculate memory usage
            element_size = input_data.element_size()
            total_memory = input_data.numel() * element_size
            
            # Check memory limits
            max_memory = context.get('max_memory_mb', 1000) * 1024 * 1024  # Convert MB to bytes
            if total_memory > max_memory:
                violations.append(f"Input requires {total_memory / 1024 / 1024:.2f} MB, exceeding limit")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'confidence': 0.85
        }
    
    def _validate_against_patterns(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against known malicious patterns."""
        
        violations = []
        
        # Convert input to string for pattern matching
        if isinstance(input_data, str):
            text_data = input_data
        elif hasattr(input_data, '__str__'):
            text_data = str(input_data)
        else:
            return {'passed': True, 'confidence': 1.0}
        
        # Check against malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, text_data, re.IGNORECASE):
                violations.append(f"Malicious pattern detected: {pattern}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'confidence': 0.7
        }
    
    def _validate_statistical_bounds(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical properties of input data."""
        
        if not isinstance(input_data, torch.Tensor) or input_data.numel() < 10:
            return {'passed': True, 'confidence': 1.0}
        
        violations = []
        
        # Calculate statistics
        data_flat = input_data.flatten().float()
        mean_val = float(torch.mean(data_flat))
        std_val = float(torch.std(data_flat))
        
        # Check for statistical anomalies
        if abs(mean_val) > 100:
            violations.append(f"Unusual mean value: {mean_val}")
        
        if std_val > 100 or std_val < 1e-8:
            violations.append(f"Unusual standard deviation: {std_val}")
        
        # Store statistics for baseline
        self.input_statistics['mean'].append(mean_val)
        self.input_statistics['std'].append(std_val)
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'confidence': 0.6
        }
    
    def _detect_injection_attacks(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential injection attacks."""
        
        violations = []
        
        # Check for command injection patterns
        if isinstance(input_data, str):
            injection_indicators = [
                ';', '&&', '||', '`', '$(',
                'rm ', 'del ', 'format ', 'drop ',
                'update ', 'insert ', 'delete '
            ]
            
            for indicator in injection_indicators:
                if indicator in input_data.lower():
                    violations.append(f"Potential injection attack detected: {indicator}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'confidence': 0.8
        }
    
    def _detect_adversarial_patterns(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect adversarial patterns using ML-based detection."""
        
        if not isinstance(input_data, torch.Tensor):
            return {'passed': True, 'confidence': 1.0}
        
        violations = []
        
        # Use anomaly detector if trained
        if self.is_trained and input_data.numel() > 0:
            try:
                # Flatten and normalize input
                data_flat = input_data.flatten().cpu().numpy().reshape(1, -1)
                
                # Limit feature size for efficiency
                if data_flat.shape[1] > 10000:
                    data_flat = data_flat[:, :10000]
                
                # Detect anomalies
                anomaly_score = self.input_anomaly_detector.decision_function(data_flat)
                
                if anomaly_score < -0.5:  # Threshold for anomaly
                    violations.append(f"Adversarial pattern detected (anomaly score: {anomaly_score[0]:.3f})")
                
            except Exception as e:
                logger.warning(f"Adversarial detection failed: {e}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'confidence': 0.6
        }
    
    def _calculate_risk_level(self, violations: List[str], confidence: float) -> str:
        """Calculate overall risk level based on violations and confidence."""
        
        if not violations:
            return 'low'
        
        # Count critical violations
        critical_keywords = ['injection', 'adversarial', 'malicious', 'extreme', 'dangerous']
        critical_count = sum(1 for v in violations if any(kw in v.lower() for kw in critical_keywords))
        
        if critical_count > 0 or confidence < 0.3:
            return 'critical'
        elif len(violations) > 3 or confidence < 0.5:
            return 'high'
        elif len(violations) > 1 or confidence < 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_input_hash(self, input_data: Any) -> str:
        """Calculate hash of input data for integrity tracking."""
        
        try:
            if isinstance(input_data, torch.Tensor):
                data_bytes = input_data.cpu().numpy().tobytes()
            else:
                data_bytes = str(input_data).encode()
            
            return hashlib.sha256(data_bytes).hexdigest()
            
        except Exception:
            return hashlib.sha256(str(input_data).encode()).hexdigest()
    
    def train_anomaly_detector(self, training_data: List[torch.Tensor]):
        """Train anomaly detector on normal input patterns."""
        
        if len(training_data) < 10:
            logger.warning("Insufficient training data for anomaly detector")
            return
        
        try:
            # Prepare training features
            features = []
            for tensor in training_data:
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    # Extract statistical features
                    flat_data = tensor.flatten().cpu().numpy()
                    
                    if len(flat_data) > 10000:
                        flat_data = flat_data[:10000]  # Limit for efficiency
                    
                    features.append(flat_data)
            
            if features:
                # Pad features to same length
                max_length = max(len(f) for f in features)
                padded_features = []
                
                for f in features:
                    if len(f) < max_length:
                        padded = np.pad(f, (0, max_length - len(f)), mode='constant')
                    else:
                        padded = f[:max_length]
                    padded_features.append(padded)
                
                # Train anomaly detector
                feature_matrix = np.array(padded_features)
                self.input_anomaly_detector.fit(feature_matrix)
                self.is_trained = True
                
                logger.info(f"Anomaly detector trained on {len(features)} samples")
        
        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {e}")


class ModelIntegrityVerifier:
    """Verify model integrity and detect tampering."""
    
    def __init__(self, config):
        self.config = config
        self.model_signatures = {}
        self.integrity_logs = deque(maxlen=1000)
        
    def create_model_signature(self, model: nn.Module, metadata: Dict[str, Any]) -> str:
        """Create cryptographic signature for model integrity."""
        
        try:
            # Get model state
            state_dict = model.state_dict()
            
            # Calculate model hash
            model_data = []
            for key, tensor in state_dict.items():
                tensor_bytes = tensor.cpu().numpy().tobytes()
                model_data.append(f"{key}:{hashlib.sha256(tensor_bytes).hexdigest()}")
            
            model_string = "|".join(sorted(model_data))
            
            # Include metadata in signature
            metadata_string = json.dumps(metadata, sort_keys=True)
            full_string = f"{model_string}|{metadata_string}"
            
            # Create signature
            signature = hashlib.sha512(full_string.encode()).hexdigest()
            
            # Store signature
            self.model_signatures[metadata.get('model_id', 'default')] = {
                'signature': signature,
                'timestamp': time.time(),
                'metadata': metadata,
                'parameter_count': sum(p.numel() for p in model.parameters())
            }
            
            logger.info(f"Model signature created: {signature[:16]}...")
            
            return signature
            
        except Exception as e:
            logger.error(f"Failed to create model signature: {e}")
            raise
    
    def verify_model_integrity(
        self, 
        model: nn.Module, 
        model_id: str,
        expected_signature: Optional[str] = None
    ) -> SecurityValidationResult:
        """Verify model integrity against stored or provided signature."""
        
        violations = []
        
        try:
            # Calculate current signature
            current_metadata = {'model_id': model_id, 'verification_time': time.time()}
            current_signature = self.create_model_signature(model, current_metadata)
            
            # Compare with expected signature
            if expected_signature:
                target_signature = expected_signature
            elif model_id in self.model_signatures:
                target_signature = self.model_signatures[model_id]['signature']
            else:
                violations.append(f"No reference signature found for model {model_id}")
                target_signature = None
            
            if target_signature and current_signature != target_signature:
                violations.append("Model signature mismatch - potential tampering detected")
                
                # Log integrity violation
                self.integrity_logs.append({
                    'model_id': model_id,
                    'timestamp': time.time(),
                    'violation_type': 'signature_mismatch',
                    'expected_signature': target_signature[:16] + "...",
                    'actual_signature': current_signature[:16] + "..."
                })
            
            # Additional integrity checks
            integrity_checks = self._perform_additional_integrity_checks(model, model_id)
            violations.extend(integrity_checks)
            
        except Exception as e:
            violations.append(f"Integrity verification failed: {str(e)}")
        
        risk_level = 'critical' if violations else 'low'
        
        return SecurityValidationResult(
            passed=len(violations) == 0,
            confidence_score=1.0 if len(violations) == 0 else 0.1,
            risk_level=risk_level,
            violations=violations,
            recommendations=self._get_integrity_recommendations(violations),
            metadata={
                'model_id': model_id,
                'verification_timestamp': time.time(),
                'current_signature': current_signature[:16] + "..."
            }
        )
    
    def _perform_additional_integrity_checks(self, model: nn.Module, model_id: str) -> List[str]:
        """Perform additional integrity checks beyond signature verification."""
        
        violations = []
        
        try:
            # Check parameter statistics
            all_params = []
            for param in model.parameters():
                if param.requires_grad:
                    all_params.extend(param.flatten().cpu().detach().numpy())
            
            if all_params:
                all_params = np.array(all_params)
                
                # Check for suspicious parameter distributions
                param_mean = np.mean(all_params)
                param_std = np.std(all_params)
                
                if abs(param_mean) > 10:
                    violations.append(f"Unusual parameter mean: {param_mean}")
                
                if param_std > 100 or param_std < 1e-8:
                    violations.append(f"Unusual parameter standard deviation: {param_std}")
                
                # Check for NaN or infinite parameters
                if np.isnan(all_params).any():
                    violations.append("Model contains NaN parameters")
                
                if np.isinf(all_params).any():
                    violations.append("Model contains infinite parameters")
            
            # Check model structure integrity
            expected_modules = ['adapters', 'classification_heads', 'task_router', 'base_model']
            for module_name in expected_modules:
                if hasattr(model, module_name):
                    module = getattr(model, module_name)
                    if module is None:
                        violations.append(f"Critical module {module_name} is None")
        
        except Exception as e:
            violations.append(f"Additional integrity check failed: {str(e)}")
        
        return violations
    
    def _get_integrity_recommendations(self, violations: List[str]) -> List[str]:
        """Get recommendations based on integrity violations."""
        
        recommendations = []
        
        for violation in violations:
            if 'signature_mismatch' in violation.lower():
                recommendations.append("Restore model from trusted checkpoint")
                recommendations.append("Investigate potential security breach")
            elif 'nan' in violation.lower():
                recommendations.append("Reset affected parameters")
                recommendations.append("Check training stability")
            elif 'infinite' in violation.lower():
                recommendations.append("Apply gradient clipping")
                recommendations.append("Reduce learning rate")
        
        return list(set(recommendations))  # Remove duplicates


class DifferentialPrivacyManager:
    """Implement differential privacy mechanisms for research compliance."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Failure probability
        self.privacy_spent = 0.0
        self.privacy_log = []
        
    def add_noise_to_gradients(
        self, 
        gradients: Dict[str, torch.Tensor], 
        sensitivity: float = 1.0,
        mechanism: str = 'gaussian'
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients."""
        
        if mechanism == 'gaussian':
            return self._gaussian_mechanism(gradients, sensitivity)
        elif mechanism == 'laplace':
            return self._laplace_mechanism(gradients, sensitivity)
        else:
            raise ValueError(f"Unknown DP mechanism: {mechanism}")
    
    def _gaussian_mechanism(self, gradients: Dict[str, torch.Tensor], sensitivity: float) -> Dict[str, torch.Tensor]:
        """Apply Gaussian mechanism for differential privacy."""
        
        noisy_gradients = {}
        
        # Calculate noise scale
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        
        for name, grad in gradients.items():
            if grad is not None:
                # Add Gaussian noise
                noise = torch.normal(0, sigma, size=grad.shape, device=grad.device)
                noisy_gradients[name] = grad + noise
            else:
                noisy_gradients[name] = grad
        
        # Update privacy budget
        self.privacy_spent += self.epsilon
        self.privacy_log.append({
            'timestamp': time.time(),
            'mechanism': 'gaussian',
            'epsilon_spent': self.epsilon,
            'sigma': sigma,
            'total_spent': self.privacy_spent
        })
        
        return noisy_gradients
    
    def _laplace_mechanism(self, gradients: Dict[str, torch.Tensor], sensitivity: float) -> Dict[str, torch.Tensor]:
        """Apply Laplace mechanism for differential privacy."""
        
        noisy_gradients = {}
        
        # Calculate noise scale
        b = sensitivity / self.epsilon
        
        for name, grad in gradients.items():
            if grad is not None:
                # Add Laplace noise
                noise = torch.distributions.Laplace(0, b).sample(grad.shape).to(grad.device)
                noisy_gradients[name] = grad + noise
            else:
                noisy_gradients[name] = grad
        
        # Update privacy budget
        self.privacy_spent += self.epsilon
        self.privacy_log.append({
            'timestamp': time.time(),
            'mechanism': 'laplace',
            'epsilon_spent': self.epsilon,
            'scale': b,
            'total_spent': self.privacy_spent
        })
        
        return noisy_gradients
    
    def get_privacy_budget_remaining(self) -> float:
        """Get remaining privacy budget."""
        
        max_budget = 10.0  # Maximum recommended privacy budget
        return max(0.0, max_budget - self.privacy_spent)
    
    def reset_privacy_budget(self):
        """Reset privacy budget (use with caution)."""
        
        logger.warning("Privacy budget reset - ensure this is appropriate for your use case")
        self.privacy_spent = 0.0
        self.privacy_log.append({
            'timestamp': time.time(),
            'action': 'budget_reset',
            'previous_spent': self.privacy_spent
        })


class SecureMultiPartyComputation:
    """Implement secure multi-party computation for distributed learning."""
    
    def __init__(self, party_id: str, num_parties: int):
        self.party_id = party_id
        self.num_parties = num_parties
        self.shared_secrets = {}
        self.computation_log = []
        
    def create_secret_shares(
        self, 
        tensor: torch.Tensor, 
        threshold: int = None
    ) -> Dict[str, torch.Tensor]:
        """Create secret shares of a tensor using Shamir's Secret Sharing."""
        
        if threshold is None:
            threshold = (self.num_parties // 2) + 1
        
        shares = {}
        
        # Generate polynomial coefficients
        coefficients = [tensor] + [torch.randn_like(tensor) for _ in range(threshold - 1)]
        
        # Evaluate polynomial at different points for each party
        for party_idx in range(1, self.num_parties + 1):
            share = coefficients[0].clone()
            
            for degree, coeff in enumerate(coefficients[1:], 1):
                share += coeff * (party_idx ** degree)
            
            shares[f"party_{party_idx}"] = share
        
        return shares
    
    def reconstruct_secret(
        self, 
        shares: Dict[str, torch.Tensor], 
        party_indices: List[int]
    ) -> torch.Tensor:
        """Reconstruct secret from shares using Lagrange interpolation."""
        
        if len(shares) < len(party_indices):
            raise ValueError("Insufficient shares for reconstruction")
        
        # Initialize result
        result = torch.zeros_like(list(shares.values())[0])
        
        # Lagrange interpolation
        for i, party_i in enumerate(party_indices):
            share_key = f"party_{party_i}"
            if share_key not in shares:
                continue
            
            # Calculate Lagrange coefficient
            lagrange_coeff = 1.0
            for j, party_j in enumerate(party_indices):
                if i != j:
                    lagrange_coeff *= -party_j / (party_i - party_j)
            
            result += shares[share_key] * lagrange_coeff
        
        return result
    
    def secure_aggregation(
        self, 
        local_update: torch.Tensor,
        other_party_shares: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Perform secure aggregation of model updates."""
        
        # Create shares of local update
        local_shares = self.create_secret_shares(local_update)
        
        # Combine with other parties' shares
        all_shares = {**local_shares, **other_party_shares}
        
        # Reconstruct aggregated result
        party_indices = list(range(1, len(all_shares) + 1))
        aggregated = self.reconstruct_secret(all_shares, party_indices)
        
        # Log computation
        self.computation_log.append({
            'timestamp': time.time(),
            'operation': 'secure_aggregation',
            'parties_involved': list(all_shares.keys()),
            'tensor_shape': list(local_update.shape)
        })
        
        return aggregated / len(all_shares)  # Average


class DataProvenanceTracker:
    """Track data provenance and lineage for research reproducibility."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path('./provenance')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.provenance_records = {}
        self.access_log = deque(maxlen=10000)
        
        # Encryption for sensitive provenance data
        self.encryption_key = self._generate_encryption_key()
        
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for provenance data."""
        
        key_file = self.storage_path / '.provenance_key'
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def create_data_record(
        self, 
        data: Any,
        source: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Create provenance record for data."""
        
        # Generate unique data ID
        data_id = hashlib.sha256(f"{source}_{time.time()}".encode()).hexdigest()[:16]
        
        # Calculate integrity hash
        if isinstance(data, torch.Tensor):
            data_bytes = data.cpu().numpy().tobytes()
        else:
            data_bytes = str(data).encode()
        
        integrity_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Create record
        record = DataProvenanceRecord(
            data_id=data_id,
            source=source,
            timestamp=time.time(),
            integrity_hash=integrity_hash
        )
        
        # Add metadata
        record.validation_results = metadata
        
        # Store record
        self.provenance_records[data_id] = record
        self._persist_record(record)
        
        logger.info(f"Data provenance record created: {data_id}")
        
        return data_id
    
    def add_transformation(
        self, 
        data_id: str, 
        transformation: str, 
        output_data: Any
    ) -> str:
        """Add transformation to data lineage."""
        
        if data_id not in self.provenance_records:
            raise ValueError(f"Data ID {data_id} not found in provenance records")
        
        # Create new record for transformed data
        new_data_id = self.create_data_record(
            output_data, 
            f"transform:{transformation}", 
            {'parent_data_id': data_id, 'transformation': transformation}
        )
        
        # Update parent record
        parent_record = self.provenance_records[data_id]
        parent_record.transformations.append(f"{transformation} -> {new_data_id}")
        
        self._persist_record(parent_record)
        
        return new_data_id
    
    def log_data_access(self, data_id: str, accessor: str, purpose: str):
        """Log data access for audit trail."""
        
        access_entry = {
            'data_id': data_id,
            'accessor': accessor,
            'purpose': purpose,
            'timestamp': time.time(),
            'ip_address': '127.0.0.1'  # Would get real IP in production
        }
        
        # Add to record
        if data_id in self.provenance_records:
            self.provenance_records[data_id].access_log.append(access_entry)
            self._persist_record(self.provenance_records[data_id])
        
        # Add to global access log
        self.access_log.append(access_entry)
    
    def verify_data_integrity(self, data_id: str, current_data: Any) -> bool:
        """Verify data integrity against stored hash."""
        
        if data_id not in self.provenance_records:
            return False
        
        record = self.provenance_records[data_id]
        
        # Calculate current hash
        if isinstance(current_data, torch.Tensor):
            data_bytes = current_data.cpu().numpy().tobytes()
        else:
            data_bytes = str(current_data).encode()
        
        current_hash = hashlib.sha256(data_bytes).hexdigest()
        
        return current_hash == record.integrity_hash
    
    def get_data_lineage(self, data_id: str) -> Dict[str, Any]:
        """Get complete lineage for data."""
        
        if data_id not in self.provenance_records:
            return {'error': f'Data ID {data_id} not found'}
        
        record = self.provenance_records[data_id]
        
        return {
            'data_id': data_id,
            'source': record.source,
            'created_timestamp': record.timestamp,
            'transformations': record.transformations,
            'access_count': len(record.access_log),
            'integrity_status': 'verified' if record.integrity_hash else 'unverified',
            'validation_results': record.validation_results
        }
    
    def _persist_record(self, record: DataProvenanceRecord):
        """Persist provenance record to disk."""
        
        try:
            # Serialize record
            record_data = json.dumps({
                'data_id': record.data_id,
                'source': record.source,
                'timestamp': record.timestamp,
                'transformations': record.transformations,
                'validation_results': record.validation_results,
                'integrity_hash': record.integrity_hash,
                'access_log': record.access_log
            })
            
            # Encrypt sensitive data
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(record_data.encode())
            
            # Write to file
            record_file = self.storage_path / f"{record.data_id}.prov"
            with open(record_file, 'wb') as f:
                f.write(encrypted_data)
                
        except Exception as e:
            logger.error(f"Failed to persist provenance record {record.data_id}: {e}")


class ComprehensiveSecurityFramework:
    """Comprehensive security framework integrating all security components."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize security components
        self.input_validator = AdvancedInputValidator(config)
        self.integrity_verifier = ModelIntegrityVerifier(config)
        self.privacy_manager = DifferentialPrivacyManager(
            epsilon=getattr(config, 'dp_epsilon', 1.0),
            delta=getattr(config, 'dp_delta', 1e-5)
        )
        
        self.provenance_tracker = DataProvenanceTracker(
            getattr(config, 'provenance_storage', None)
        )
        
        # Security audit log
        self.security_log = deque(maxlen=10000)
        self.threat_detection_active = True
        
        # Thread pool for async security operations
        self.security_executor = ThreadPoolExecutor(max_workers=4)
        
    def comprehensive_security_check(
        self, 
        model: nn.Module,
        input_data: Any,
        context: Dict[str, Any]
    ) -> SecurityValidationResult:
        """Perform comprehensive security validation."""
        
        all_violations = []
        all_recommendations = []
        confidence_scores = []
        
        try:
            # Input validation
            input_result = self.input_validator.validate_input(input_data, context)
            all_violations.extend(input_result.violations)
            all_recommendations.extend(input_result.recommendations)
            confidence_scores.append(input_result.confidence_score)
            
            # Model integrity verification
            model_id = context.get('model_id', 'default')
            integrity_result = self.integrity_verifier.verify_model_integrity(model, model_id)
            all_violations.extend(integrity_result.violations)
            all_recommendations.extend(integrity_result.recommendations)
            confidence_scores.append(integrity_result.confidence_score)
            
            # Additional security checks
            if self.threat_detection_active:
                threat_violations = self._detect_advanced_threats(model, input_data, context)
                all_violations.extend(threat_violations)
            
            # Log security check
            self._log_security_event('comprehensive_check', {
                'violations_count': len(all_violations),
                'confidence_score': np.mean(confidence_scores) if confidence_scores else 0.0,
                'model_id': model_id,
                'input_type': type(input_data).__name__
            })
            
        except Exception as e:
            all_violations.append(f"Security framework error: {str(e)}")
            logger.error(f"Security check failed: {e}")
        
        # Calculate overall risk assessment
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        risk_level = self._assess_overall_risk(all_violations, overall_confidence)
        
        return SecurityValidationResult(
            passed=len(all_violations) == 0,
            confidence_score=overall_confidence,
            risk_level=risk_level,
            violations=all_violations,
            recommendations=list(set(all_recommendations)),  # Remove duplicates
            metadata={
                'security_framework_version': '1.0',
                'check_timestamp': time.time(),
                'components_checked': ['input_validation', 'model_integrity', 'threat_detection']
            }
        )
    
    def _detect_advanced_threats(
        self, 
        model: nn.Module, 
        input_data: Any, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Detect advanced security threats."""
        
        threats = []
        
        try:
            # Model extraction attack detection
            if self._detect_model_extraction_attempt(context):
                threats.append("Potential model extraction attack detected")
            
            # Membership inference attack detection
            if self._detect_membership_inference_attack(input_data, context):
                threats.append("Potential membership inference attack detected")
            
            # Adversarial input detection (advanced)
            if self._detect_advanced_adversarial_input(input_data, model):
                threats.append("Advanced adversarial input pattern detected")
                
        except Exception as e:
            logger.error(f"Advanced threat detection failed: {e}")
        
        return threats
    
    def _detect_model_extraction_attempt(self, context: Dict[str, Any]) -> bool:
        """Detect potential model extraction attacks."""
        
        # Check for suspicious query patterns
        query_count = context.get('recent_query_count', 0)
        query_diversity = context.get('query_diversity', 1.0)
        
        # High query count with low diversity might indicate extraction
        return query_count > 1000 and query_diversity < 0.3
    
    def _detect_membership_inference_attack(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """Detect potential membership inference attacks."""
        
        # Check for patterns indicative of membership inference
        confidence_variance = context.get('prediction_confidence_variance', 0.0)
        
        # Unusual confidence patterns might indicate membership inference
        return confidence_variance > 0.8
    
    def _detect_advanced_adversarial_input(self, input_data: Any, model: nn.Module) -> bool:
        """Detect advanced adversarial inputs using gradient analysis."""
        
        if not isinstance(input_data, torch.Tensor) or not input_data.requires_grad:
            return False
        
        try:
            # Calculate input gradients
            input_data.requires_grad_(True)
            
            # Simple forward pass (would be more sophisticated in practice)
            if hasattr(model, 'base_model'):
                output = model.base_model(input_data.unsqueeze(0) if input_data.dim() == 1 else input_data)
            else:
                return False
            
            # Calculate gradient magnitude
            grad = torch.autograd.grad(output.sum(), input_data, create_graph=False)[0]
            grad_magnitude = torch.norm(grad)
            
            # Threshold for adversarial detection
            return grad_magnitude > 10.0
            
        except Exception:
            return False
    
    def _assess_overall_risk(self, violations: List[str], confidence: float) -> str:
        """Assess overall risk level from all security checks."""
        
        if not violations and confidence > 0.8:
            return 'low'
        
        critical_keywords = [
            'extraction', 'inference', 'adversarial', 'tampering',
            'injection', 'malicious', 'critical', 'dangerous'
        ]
        
        critical_violations = sum(
            1 for v in violations 
            if any(keyword in v.lower() for keyword in critical_keywords)
        )
        
        if critical_violations > 0 or confidence < 0.3:
            return 'critical'
        elif len(violations) > 5 or confidence < 0.5:
            return 'high'
        elif len(violations) > 2 or confidence < 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _log_security_event(self, event_type: str, metadata: Dict[str, Any]):
        """Log security events for audit purposes."""
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'metadata': metadata,
            'threat_detection_active': self.threat_detection_active
        }
        
        self.security_log.append(event)
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        recent_events = list(self.security_log)[-100:]  # Last 100 events
        
        event_counts = defaultdict(int)
        violation_patterns = defaultdict(int)
        
        for event in recent_events:
            event_counts[event['event_type']] += 1
            
            if 'violations_count' in event['metadata']:
                violation_patterns[event['metadata']['violations_count']] += 1
        
        return {
            'report_timestamp': time.time(),
            'security_framework_status': 'active',
            'recent_event_summary': dict(event_counts),
            'violation_pattern_analysis': dict(violation_patterns),
            'privacy_budget_remaining': self.privacy_manager.get_privacy_budget_remaining(),
            'threat_detection_active': self.threat_detection_active,
            'integrity_verifications': len(self.integrity_verifier.model_signatures),
            'provenance_records': len(self.provenance_tracker.provenance_records),
            'security_recommendations': self._generate_security_recommendations()
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on recent activity."""
        
        recommendations = []
        
        # Check privacy budget
        if self.privacy_manager.get_privacy_budget_remaining() < 2.0:
            recommendations.append("Consider resetting privacy budget or implementing budget management")
        
        # Check threat detection
        if not self.threat_detection_active:
            recommendations.append("Enable advanced threat detection for enhanced security")
        
        # Check recent violations
        recent_violations = sum(
            event['metadata'].get('violations_count', 0)
            for event in list(self.security_log)[-50:]  # Last 50 events
        )
        
        if recent_violations > 10:
            recommendations.append("High violation count detected - review security policies")
        
        return recommendations