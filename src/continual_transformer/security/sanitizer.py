"""Data sanitization and input cleaning for continual learning models."""

import re
import html
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)

class DataSanitizer:
    """Comprehensive data sanitization for continual learning inputs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_length = self.config.get("max_sequence_length", 512)
        self.preserve_case = self.config.get("preserve_case", True)
        self.remove_urls = self.config.get("remove_urls", False)
        self.remove_emails = self.config.get("remove_emails", False)
        
        # Compile regex patterns for efficiency
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self._html_tag_pattern = re.compile(r'<[^>]+>')
        self._excessive_whitespace = re.compile(r'\s{3,}')
        self._special_chars = re.compile(r'[^\w\s.,!?;:()\'"/-]')
        self._repeated_chars = re.compile(r'(.)\1{4,}')  # 5+ repeated characters
        
        # Dangerous patterns to remove
        self._dangerous_patterns = [
            re.compile(r'<script.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
        ]
        
        # Common profanity filter (basic implementation)
        self._profanity_pattern = None
        if self.config.get("filter_profanity", False):
            profanity_words = self.config.get("profanity_list", [])
            if profanity_words:
                pattern_str = r'\b(' + '|'.join(re.escape(word) for word in profanity_words) + r')\b'
                self._profanity_pattern = re.compile(pattern_str, re.IGNORECASE)
    
    def sanitize_text(
        self, 
        text: Union[str, List[str]], 
        strict_mode: bool = False
    ) -> Union[str, List[str]]:
        """Sanitize text input by removing dangerous content and normalizing.
        
        Args:
            text: Input text or list of texts
            strict_mode: Whether to apply aggressive sanitization
            
        Returns:
            Sanitized text in same format as input
        """
        is_list = isinstance(text, list)
        texts = text if is_list else [text]
        
        sanitized_texts = []
        for txt in texts:
            sanitized = self._sanitize_single_text(txt, strict_mode)
            sanitized_texts.append(sanitized)
        
        return sanitized_texts if is_list else sanitized_texts[0]
    
    def _sanitize_single_text(self, text: str, strict_mode: bool) -> str:
        """Sanitize a single text string."""
        if not isinstance(text, str):
            text = str(text)
        
        original_length = len(text)
        
        # Step 1: Remove dangerous patterns
        for pattern in self._dangerous_patterns:
            text = pattern.sub('', text)
        
        # Step 2: HTML decoding and tag removal
        text = html.unescape(text)
        text = self._html_tag_pattern.sub(' ', text)
        
        # Step 3: URL/Email removal (if configured)
        if self.remove_urls:
            text = self._url_pattern.sub('[URL]', text)
        
        if self.remove_emails:
            text = self._email_pattern.sub('[EMAIL]', text)
        
        # Step 4: Profanity filtering (if enabled)
        if self._profanity_pattern:
            text = self._profanity_pattern.sub('[FILTERED]', text)
        
        # Step 5: Character-level sanitization
        if strict_mode:
            # Remove excessive special characters
            text = self._special_chars.sub('', text)
        else:
            # Replace unusual special characters with spaces
            text = re.sub(r'[^\w\s.,!?;:()\'"/-]', ' ', text)
        
        # Step 6: Normalize repeated characters
        text = self._repeated_chars.sub(r'\1\1\1', text)  # Max 3 repetitions
        
        # Step 7: Whitespace normalization
        text = self._excessive_whitespace.sub(' ', text)
        text = text.strip()
        
        # Step 8: Length validation
        if len(text) > self.max_length * 5:  # Allow some buffer over model max
            text = text[:self.max_length * 5]
            logger.warning(f"Text truncated from {original_length} to {len(text)} characters")
        
        # Step 9: Case normalization (if configured)
        if not self.preserve_case and strict_mode:
            text = text.lower()
        
        # Step 10: Final validation
        if not text.strip():
            text = "[EMPTY]"  # Provide fallback for empty text
        
        return text
    
    def sanitize_batch(
        self, 
        batch: Dict[str, Any], 
        text_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Sanitize a batch of data.
        
        Args:
            batch: Data batch dictionary
            text_keys: Keys containing text data to sanitize
            
        Returns:
            Sanitized batch
        """
        if text_keys is None:
            text_keys = ['text', 'input_text', 'sentence', 'content']
        
        sanitized_batch = batch.copy()
        
        for key in text_keys:
            if key in batch:
                try:
                    original_data = batch[key]
                    sanitized_data = self.sanitize_text(original_data)
                    sanitized_batch[key] = sanitized_data
                    
                    # Log significant changes
                    if isinstance(original_data, str) and len(original_data) != len(sanitized_data):
                        logger.info(f"Sanitized '{key}': {len(original_data)} -> {len(sanitized_data)} chars")
                        
                except Exception as e:
                    logger.error(f"Failed to sanitize key '{key}': {e}")
                    # Keep original data on sanitization failure
                    sanitized_batch[key] = batch[key]
        
        return sanitized_batch

class InputSanitizer:
    """Specialized sanitizer for model inputs including tensors."""
    
    def __init__(self):
        self.data_sanitizer = DataSanitizer()
    
    def sanitize_tensor(
        self, 
        tensor: torch.Tensor, 
        clip_range: Optional[Tuple[float, float]] = None,
        replace_nan: bool = True,
        replace_inf: bool = True
    ) -> torch.Tensor:
        """Sanitize tensor inputs by handling NaN, inf, and extreme values.
        
        Args:
            tensor: Input tensor
            clip_range: Optional range to clip values (min, max)
            replace_nan: Whether to replace NaN values
            replace_inf: Whether to replace infinite values
            
        Returns:
            Sanitized tensor
        """
        if not isinstance(tensor, torch.Tensor):
            logger.warning("Input is not a tensor, returning unchanged")
            return tensor
        
        sanitized = tensor.clone()
        modifications_made = False
        
        # Handle NaN values
        if replace_nan and torch.isnan(sanitized).any():
            nan_count = torch.isnan(sanitized).sum().item()
            sanitized = torch.where(torch.isnan(sanitized), torch.zeros_like(sanitized), sanitized)
            modifications_made = True
            logger.warning(f"Replaced {nan_count} NaN values with zeros")
        
        # Handle infinite values
        if replace_inf and torch.isinf(sanitized).any():
            inf_count = torch.isinf(sanitized).sum().item()
            
            # Replace positive infinity with large finite value
            pos_inf_mask = torch.isposinf(sanitized)
            if pos_inf_mask.any():
                max_finite = torch.finfo(sanitized.dtype).max / 2
                sanitized = torch.where(pos_inf_mask, torch.full_like(sanitized, max_finite), sanitized)
            
            # Replace negative infinity with large negative finite value
            neg_inf_mask = torch.isneginf(sanitized)
            if neg_inf_mask.any():
                min_finite = torch.finfo(sanitized.dtype).min / 2
                sanitized = torch.where(neg_inf_mask, torch.full_like(sanitized, min_finite), sanitized)
            
            modifications_made = True
            logger.warning(f"Replaced {inf_count} infinite values")
        
        # Clip extreme values if range provided
        if clip_range is not None:
            min_val, max_val = clip_range
            original_min = sanitized.min().item()
            original_max = sanitized.max().item()
            
            sanitized = torch.clamp(sanitized, min_val, max_val)
            
            if original_min < min_val or original_max > max_val:
                modifications_made = True
                logger.info(f"Clipped tensor values to range [{min_val}, {max_val}]")
        
        if modifications_made:
            logger.debug("Tensor sanitization completed with modifications")
        
        return sanitized
    
    def sanitize_model_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        vocab_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sanitize complete model input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels (if provided)
            vocab_size: Vocabulary size for validation
            
        Returns:
            Tuple of sanitized (input_ids, attention_mask, labels)
        """
        # Sanitize input_ids
        sanitized_input_ids = self.sanitize_tensor(
            input_ids, 
            clip_range=(0, vocab_size-1) if vocab_size else None,
            replace_nan=True,
            replace_inf=True
        )
        
        # Ensure input_ids are integers
        if sanitized_input_ids.dtype != torch.long:
            sanitized_input_ids = sanitized_input_ids.long()
            logger.info("Converted input_ids to long dtype")
        
        # Sanitize attention mask
        sanitized_attention_mask = attention_mask
        if attention_mask is not None:
            sanitized_attention_mask = self.sanitize_tensor(
                attention_mask,
                clip_range=(0, 1),
                replace_nan=True,
                replace_inf=True
            )
            
            # Ensure attention mask is binary
            sanitized_attention_mask = (sanitized_attention_mask > 0.5).long()
        
        # Sanitize labels
        sanitized_labels = labels
        if labels is not None:
            sanitized_labels = self.sanitize_tensor(
                labels,
                replace_nan=True,
                replace_inf=True
            )
            
            # Ensure labels are integers
            if sanitized_labels.dtype != torch.long:
                sanitized_labels = sanitized_labels.long()
                logger.info("Converted labels to long dtype")
            
            # Check for negative labels
            if (sanitized_labels < 0).any():
                logger.warning("Found negative labels, clipping to 0")
                sanitized_labels = torch.clamp(sanitized_labels, min=0)
        
        return sanitized_input_ids, sanitized_attention_mask, sanitized_labels
    
    def sanitize_prediction_input(
        self, 
        text: Union[str, List[str]], 
        task_id: str,
        strict_mode: bool = False
    ) -> Tuple[Union[str, List[str]], str, List[str]]:
        """Sanitize inputs for prediction.
        
        Args:
            text: Input text or list of texts
            task_id: Task identifier
            strict_mode: Whether to apply strict sanitization
            
        Returns:
            Tuple of (sanitized_text, sanitized_task_id, warnings)
        """
        warnings = []
        
        # Sanitize text
        try:
            sanitized_text = self.data_sanitizer.sanitize_text(text, strict_mode)
            
            # Check for significant changes
            if isinstance(text, str) and isinstance(sanitized_text, str):
                if len(text) != len(sanitized_text):
                    warnings.append(f"Text length changed: {len(text)} -> {len(sanitized_text)}")
            elif isinstance(text, list) and isinstance(sanitized_text, list):
                for i, (orig, clean) in enumerate(zip(text, sanitized_text)):
                    if len(orig) != len(clean):
                        warnings.append(f"Text {i} length changed: {len(orig)} -> {len(clean)}")
                        
        except Exception as e:
            warnings.append(f"Text sanitization failed: {e}")
            sanitized_text = text
        
        # Sanitize task_id
        try:
            sanitized_task_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(task_id))
            if sanitized_task_id != task_id:
                warnings.append(f"Task ID sanitized: '{task_id}' -> '{sanitized_task_id}'")
        except Exception as e:
            warnings.append(f"Task ID sanitization failed: {e}")
            sanitized_task_id = task_id
        
        return sanitized_text, sanitized_task_id, warnings

class AdvancedSanitizer:
    """Advanced sanitization with ML-based content filtering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_sanitizer = DataSanitizer(config)
        self.input_sanitizer = InputSanitizer()
        
        # Content analysis cache
        self._content_cache = {}
        self._cache_size_limit = 1000
    
    def analyze_content_safety(self, text: str) -> Dict[str, Any]:
        """Analyze content for safety issues using heuristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Safety analysis report
        """
        # Check cache first
        text_hash = hash(text)
        if text_hash in self._content_cache:
            return self._content_cache[text_hash]
        
        analysis = {
            "safe": True,
            "confidence": 1.0,
            "issues": [],
            "severity": "low",
            "recommendations": []
        }
        
        # Length-based analysis
        if len(text) > 10000:
            analysis["issues"].append("Unusually long text")
            analysis["severity"] = "medium"
        
        # Pattern-based analysis
        suspicious_patterns = [
            (r'\b(password|secret|key|token)\s*[:=]\s*\S+', "Potential credential exposure"),
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "Potential credit card number"),
            (r'\b\d{3}-\d{2}-\d{4}\b', "Potential SSN pattern"),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email address detected"),
        ]
        
        for pattern, issue in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                analysis["issues"].append(issue)
                if "credential" in issue.lower() or "credit card" in issue.lower():
                    analysis["severity"] = "high"
                    analysis["safe"] = False
                elif analysis["severity"] == "low":
                    analysis["severity"] = "medium"
        
        # Repetition analysis
        words = text.split()
        if len(words) > 10:
            word_count = {}
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
            
            max_repetitions = max(word_count.values())
            if max_repetitions > len(words) * 0.3:  # More than 30% repetition
                analysis["issues"].append("High word repetition detected")
                if analysis["severity"] == "low":
                    analysis["severity"] = "medium"
        
        # Generate recommendations
        if analysis["severity"] == "high":
            analysis["recommendations"] = ["Reject content", "Manual review required"]
        elif analysis["severity"] == "medium":
            analysis["recommendations"] = ["Apply sanitization", "Monitor usage"]
        else:
            analysis["recommendations"] = ["Content appears safe"]
        
        # Update cache
        if len(self._content_cache) >= self._cache_size_limit:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._content_cache.keys())[:100]
            for key in oldest_keys:
                del self._content_cache[key]
        
        self._content_cache[text_hash] = analysis
        return analysis
    
    def comprehensive_sanitization(
        self,
        data: Dict[str, Any],
        safety_threshold: float = 0.8
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform comprehensive sanitization with safety analysis.
        
        Args:
            data: Input data dictionary
            safety_threshold: Minimum safety score to accept content
            
        Returns:
            Tuple of (sanitized_data, sanitization_report)
        """
        report = {
            "timestamp": "now",  # Simplified timestamp
            "safety_analyses": {},
            "sanitization_actions": [],
            "overall_safety": "unknown",
            "rejected_content": []
        }
        
        sanitized_data = data.copy()
        
        # Analyze and sanitize text fields
        text_fields = ['text', 'input_text', 'content', 'message']
        
        for field in text_fields:
            if field in data:
                try:
                    original_content = data[field]
                    
                    if isinstance(original_content, str):
                        # Safety analysis
                        safety_analysis = self.analyze_content_safety(original_content)
                        report["safety_analyses"][field] = safety_analysis
                        
                        if not safety_analysis["safe"]:
                            report["rejected_content"].append(field)
                            sanitized_data[field] = "[CONTENT_FILTERED]"
                            report["sanitization_actions"].append(f"Rejected {field} due to safety concerns")
                        else:
                            # Apply sanitization
                            sanitized_content = self.data_sanitizer.sanitize_text(original_content)
                            sanitized_data[field] = sanitized_content
                            
                            if sanitized_content != original_content:
                                report["sanitization_actions"].append(f"Sanitized {field}")
                    
                    elif isinstance(original_content, list):
                        # Handle list of texts
                        sanitized_list = []
                        for i, item in enumerate(original_content):
                            if isinstance(item, str):
                                safety_analysis = self.analyze_content_safety(item)
                                report["safety_analyses"][f"{field}[{i}]"] = safety_analysis
                                
                                if not safety_analysis["safe"]:
                                    sanitized_list.append("[CONTENT_FILTERED]")
                                    report["sanitization_actions"].append(f"Rejected {field}[{i}]")
                                else:
                                    sanitized_item = self.data_sanitizer.sanitize_text(item)
                                    sanitized_list.append(sanitized_item)
                                    
                                    if sanitized_item != item:
                                        report["sanitization_actions"].append(f"Sanitized {field}[{i}]")
                            else:
                                sanitized_list.append(item)
                        
                        sanitized_data[field] = sanitized_list
                        
                except Exception as e:
                    error_msg = f"Failed to sanitize {field}: {e}"
                    report["sanitization_actions"].append(error_msg)
                    logger.error(error_msg)
        
        # Determine overall safety
        safety_scores = [
            analysis.get("confidence", 0) for analysis in report["safety_analyses"].values()
            if analysis.get("safe", True)
        ]
        
        if not safety_scores:
            report["overall_safety"] = "unknown"
        elif min(safety_scores) >= safety_threshold:
            report["overall_safety"] = "safe"
        elif any(not analysis.get("safe", True) for analysis in report["safety_analyses"].values()):
            report["overall_safety"] = "unsafe"
        else:
            report["overall_safety"] = "caution"
        
        return sanitized_data, report

__all__ = ["DataSanitizer", "InputSanitizer", "AdvancedSanitizer"]