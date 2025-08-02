"""Structured logging configuration for continual transformer."""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging when structlog is not available."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class MLLogger:
    """Machine Learning specific logger with structured logging."""
    
    def __init__(self, name: str = "continual_transformer"):
        self.name = name
        if HAS_STRUCTLOG:
            self.logger = structlog.get_logger(name)
        else:
            self.logger = logging.getLogger(name)
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log with additional context."""
        if HAS_STRUCTLOG:
            getattr(self.logger, level)(message, **kwargs)
        else:
            # Create extra fields for standard logging
            extra = {'extra_fields': kwargs}
            getattr(self.logger, level)(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log_with_context('error', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context('debug', message, **kwargs)
    
    def log_training_start(self, task_id: str, config: Dict[str, Any]):
        """Log training start event."""
        self.info(
            "Training started",
            event_type="training_start",
            task_id=task_id,
            config=config,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_training_progress(self, task_id: str, epoch: int, step: int,
                            loss: float, accuracy: float, learning_rate: float = None):
        """Log training progress."""
        log_data = {
            "event_type": "training_progress",
            "task_id": task_id,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy
        }
        
        if learning_rate is not None:
            log_data["learning_rate"] = learning_rate
        
        self.info("Training progress", **log_data)
    
    def log_training_completion(self, task_id: str, final_metrics: Dict[str, float],
                              training_time: float):
        """Log training completion."""
        self.info(
            "Training completed",
            event_type="training_completion",
            task_id=task_id,
            metrics=final_metrics,
            training_time_seconds=training_time
        )
    
    def log_evaluation_results(self, task_id: str, metrics: Dict[str, float],
                             dataset_name: str = None):
        """Log evaluation results."""
        log_data = {
            "event_type": "evaluation",
            "task_id": task_id,
            "metrics": metrics
        }
        
        if dataset_name:
            log_data["dataset"] = dataset_name
            
        self.info("Evaluation completed", **log_data)
    
    def log_model_checkpoint(self, task_id: str, checkpoint_path: str,
                           epoch: int, metrics: Dict[str, float] = None):
        """Log model checkpoint save."""
        log_data = {
            "event_type": "checkpoint_saved",
            "task_id": task_id,
            "checkpoint_path": checkpoint_path,
            "epoch": epoch
        }
        
        if metrics:
            log_data["metrics"] = metrics
            
        self.info("Model checkpoint saved", **log_data)
    
    def log_inference(self, task_id: str, input_size: int, inference_time: float,
                     confidence: float = None):
        """Log inference event."""
        log_data = {
            "event_type": "inference",
            "task_id": task_id,
            "input_size": input_size,
            "inference_time_ms": inference_time * 1000
        }
        
        if confidence is not None:
            log_data["confidence"] = confidence
            
        self.info("Inference completed", **log_data)
    
    def log_error(self, error_type: str, error_message: str, task_id: str = None,
                 exception: Exception = None):
        """Log error with context."""
        log_data = {
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message
        }
        
        if task_id:
            log_data["task_id"] = task_id
        
        if exception:
            log_data["exception_type"] = type(exception).__name__
            log_data["exception_details"] = str(exception)
        
        self.error("Error occurred", **log_data)
    
    def log_memory_usage(self, device: str, allocated_mb: float, reserved_mb: float = None):
        """Log memory usage."""
        log_data = {
            "event_type": "memory_usage",
            "device": device,
            "allocated_mb": allocated_mb
        }
        
        if reserved_mb is not None:
            log_data["reserved_mb"] = reserved_mb
            
        self.debug("Memory usage", **log_data)
    
    def log_performance_metrics(self, task_id: str, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.info(
            "Performance metrics",
            event_type="performance_metrics",
            task_id=task_id,
            metrics=metrics
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_structlog: bool = True,
    log_format: str = "json"
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        enable_structlog: Whether to use structlog if available
        log_format: Log format ('json' or 'text')
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    if HAS_STRUCTLOG and enable_structlog:
        # Configure structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        if log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.processors.KeyValueRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        logging.basicConfig(
            level=numeric_level,
            format="%(message)s",
            handlers=[]
        )
    else:
        # Use standard logging
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if log_format == "json":
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        handlers.append(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            if log_format == "json":
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=numeric_level,
            handlers=handlers,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Add additional handlers if specified via environment
    if log_file := os.getenv('LOG_FILE'):
        file_handler = logging.FileHandler(log_file)
        if log_format == "json":
            file_handler.setFormatter(JSONFormatter())
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str = "continual_transformer") -> MLLogger:
    """Get ML logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        MLLogger instance
    """
    return MLLogger(name)


# Configure logging based on environment variables
def configure_from_env():
    """Configure logging from environment variables."""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE')
    log_format = os.getenv('LOG_FORMAT', 'json')
    
    setup_logging(
        level=log_level,
        log_file=log_file,
        log_format=log_format
    )


# Auto-configure if this module is imported
if __name__ != "__main__":
    configure_from_env()