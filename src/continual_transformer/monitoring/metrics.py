"""Prometheus metrics collection for continual transformer."""

import time
import torch
import threading
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, start_http_server
from contextlib import contextmanager


class ModelMetrics:
    """Prometheus metrics for continual learning model."""
    
    def __init__(self):
        # Training metrics
        self.training_loss = Gauge(
            'continual_transformer_training_loss',
            'Current training loss'
        )
        
        self.training_accuracy = Gauge(
            'continual_transformer_training_accuracy', 
            'Current training accuracy'
        )
        
        self.tasks_learned = Counter(
            'continual_transformer_tasks_learned_total',
            'Total number of tasks learned'
        )
        
        self.training_epochs = Counter(
            'continual_transformer_training_epochs_total',
            'Total training epochs completed',
            ['task_id']
        )
        
        # Performance metrics
        self.inference_duration = Histogram(
            'continual_transformer_inference_duration_seconds',
            'Time spent on inference',
            buckets=[0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
        )
        
        self.training_step_duration = Histogram(
            'continual_transformer_training_step_duration_seconds',
            'Time spent on training step',
            buckets=[0.01, 0.1, 1.0, 5.0, 10.0, 30.0]
        )
        
        self.batch_processing_time = Summary(
            'continual_transformer_batch_processing_seconds',
            'Time spent processing training batches'
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'continual_transformer_memory_usage_bytes',
            'Memory usage in bytes',
            ['device', 'type']
        )
        
        self.gpu_utilization = Gauge(
            'continual_transformer_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )
        
        # Model metrics
        self.model_parameters = Gauge(
            'continual_transformer_model_parameters_total',
            'Total model parameters'
        )
        
        self.trainable_parameters = Gauge(
            'continual_transformer_trainable_parameters_total',
            'Total trainable parameters'
        )
        
        self.knowledge_retention = Gauge(
            'continual_transformer_knowledge_retention_score',
            'Knowledge retention score for task',
            ['task_id']
        )
        
        # Task-specific metrics
        self.task_performance = Gauge(
            'continual_transformer_task_performance',
            'Performance score for specific task',
            ['task_id', 'metric_type']
        )
        
        self.catastrophic_forgetting = Gauge(
            'continual_transformer_catastrophic_forgetting_score',
            'Catastrophic forgetting score',
            ['task_id']
        )
        
        # System metrics
        self.model_load_time = Histogram(
            'continual_transformer_model_load_duration_seconds',
            'Time to load model',
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.data_loading_time = Histogram(
            'continual_transformer_data_loading_duration_seconds',
            'Time to load training data',
            buckets=[0.1, 1.0, 5.0, 10.0, 30.0]
        )
        
        # Error metrics
        self.training_errors = Counter(
            'continual_transformer_training_errors_total',
            'Total training errors',
            ['error_type']
        )
        
        self.inference_errors = Counter(
            'continual_transformer_inference_errors_total',
            'Total inference errors',
            ['error_type']
        )
        
        # Model info
        self.model_info = Info(
            'continual_transformer_model_info',
            'Model configuration information'
        )
        
        # Start time
        self.start_time = time.time()
        self.uptime = Gauge(
            'continual_transformer_uptime_seconds',
            'Uptime in seconds'
        )
        
        # Start background thread to update uptime
        self._start_uptime_updater()
    
    def record_training_step(self, loss: float, accuracy: float, task_id: str = None):
        """Record training step metrics."""
        self.training_loss.set(loss)
        self.training_accuracy.set(accuracy)
        
        if task_id:
            self.training_epochs.labels(task_id=task_id).inc()
    
    def record_task_completion(self, task_id: str, retention_score: float, 
                             performance_metrics: Dict[str, float] = None):
        """Record task completion metrics."""
        self.tasks_learned.inc()
        self.knowledge_retention.labels(task_id=task_id).set(retention_score)
        
        if performance_metrics:
            for metric_name, value in performance_metrics.items():
                self.task_performance.labels(
                    task_id=task_id, 
                    metric_type=metric_name
                ).set(value)
    
    def record_catastrophic_forgetting(self, task_id: str, forgetting_score: float):
        """Record catastrophic forgetting metric."""
        self.catastrophic_forgetting.labels(task_id=task_id).set(forgetting_score)
    
    @contextmanager
    def time_inference(self):
        """Context manager to time inference operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.inference_duration.observe(duration)
    
    @contextmanager  
    def time_training_step(self):
        """Context manager to time training steps."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.training_step_duration.observe(duration)
    
    @contextmanager
    def time_batch_processing(self):
        """Context manager to time batch processing."""
        with self.batch_processing_time.time():
            yield
    
    def update_memory_usage(self):
        """Update memory usage metrics."""
        # CPU memory
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.memory_usage.labels(device='cpu', type='rss').set(memory_info.rss)
        self.memory_usage.labels(device='cpu', type='vms').set(memory_info.vms)
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                
                self.memory_usage.labels(
                    device=f'gpu_{i}', 
                    type='allocated'
                ).set(allocated)
                
                self.memory_usage.labels(
                    device=f'gpu_{i}', 
                    type='reserved'
                ).set(reserved)
    
    def update_model_info(self, model, config: Dict[str, Any] = None):
        """Update model information metrics."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.model_parameters.set(total_params)
        self.trainable_parameters.set(trainable_params)
        
        # Update model info
        info_dict = {
            'total_parameters': str(total_params),
            'trainable_parameters': str(trainable_params),
            'model_type': model.__class__.__name__
        }
        
        if config:
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    info_dict[f'config_{key}'] = str(value)
        
        self.model_info.info(info_dict)
    
    def record_error(self, error_type: str, is_training: bool = True):
        """Record an error occurrence."""
        if is_training:
            self.training_errors.labels(error_type=error_type).inc()
        else:
            self.inference_errors.labels(error_type=error_type).inc()
    
    @contextmanager
    def time_model_loading(self):
        """Context manager to time model loading."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.model_load_time.observe(duration)
    
    @contextmanager
    def time_data_loading(self):
        """Context manager to time data loading."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.data_loading_time.observe(duration)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values as dictionary."""
        return {
            'training_loss': self.training_loss._value._value if hasattr(self.training_loss._value, '_value') else 0,
            'training_accuracy': self.training_accuracy._value._value if hasattr(self.training_accuracy._value, '_value') else 0,
            'tasks_learned': self.tasks_learned._value._value if hasattr(self.tasks_learned._value, '_value') else 0,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def _start_uptime_updater(self):
        """Start background thread to update uptime metric."""
        def update_uptime():
            while True:
                self.uptime.set(time.time() - self.start_time)
                time.sleep(30)  # Update every 30 seconds
        
        thread = threading.Thread(target=update_uptime, daemon=True)
        thread.start()


class MetricsServer:
    """HTTP server for Prometheus metrics."""
    
    def __init__(self, port: int = 8000, metrics: Optional[ModelMetrics] = None):
        self.port = port
        self.metrics = metrics or ModelMetrics()
        self.server_started = False
    
    def start_server(self):
        """Start Prometheus metrics HTTP server."""
        if not self.server_started:
            start_http_server(self.port)
            self.server_started = True
            print(f"Metrics server started on port {self.port}")
            print(f"Metrics available at: http://localhost:{self.port}/metrics")
    
    def get_metrics(self) -> ModelMetrics:
        """Get metrics instance."""
        return self.metrics


# Global metrics instance
_global_metrics = None

def get_global_metrics() -> ModelMetrics:
    """Get global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = ModelMetrics()
    return _global_metrics


def start_metrics_server(port: int = 8000) -> MetricsServer:
    """Start metrics server with global metrics."""
    metrics = get_global_metrics()
    server = MetricsServer(port=port, metrics=metrics)
    server.start_server()
    return server