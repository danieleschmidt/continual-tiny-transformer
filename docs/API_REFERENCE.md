# 📚 API Reference Guide

## Overview

The Continual Tiny Transformer provides a comprehensive API for zero-parameter continual learning. This guide covers all available endpoints, classes, and methods.

## 🚀 Quick Start

```python
from continual_transformer.api import ContinualLearningAPI

# Initialize API
api = ContinualLearningAPI(
    model_name="distilbert-base-uncased",
    max_tasks=50,
    device="cuda"  # or "cpu"
)

# Add a task
api.add_task("sentiment", num_labels=2)

# Train the task
train_data = [
    {"text": "Great product!", "label": 1},
    {"text": "Terrible service", "label": 0}
]
metrics = api.train_task("sentiment", train_data, epochs=10)

# Make predictions
result = api.predict("Amazing quality!", "sentiment")
print(result)  # {'predictions': [1], 'probabilities': [[0.2, 0.8]]}
```

## 🏗️ Core API Classes

### ContinualLearningAPI

The main entry point for all continual learning operations.

#### Constructor

```python
ContinualLearningAPI(
    model_name: str = "distilbert-base-uncased",
    max_tasks: int = 50,
    device: Optional[str] = None
)
```

**Parameters:**
- `model_name`: Pre-trained model name from Hugging Face
- `max_tasks`: Maximum number of tasks to support
- `device`: Device to use ("cuda", "cpu", or None for auto-detection)

#### Methods

##### add_task()

```python
def add_task(
    task_id: str, 
    num_labels: int, 
    task_type: str = "classification"
) -> None
```

Register a new task with the model.

**Parameters:**
- `task_id`: Unique identifier for the task
- `num_labels`: Number of output classes
- `task_type`: Type of task ("classification", "regression")

**Example:**
```python
api.add_task("emotion", num_labels=6, task_type="classification")
```

##### train_task()

```python
def train_task(
    task_id: str,
    train_data: List[Dict[str, Any]],
    eval_data: Optional[List[Dict[str, Any]]] = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> Dict[str, float]
```

Train the model on a specific task.

**Parameters:**
- `task_id`: Task identifier
- `train_data`: Training data as list of {"text": str, "label": int} dicts
- `eval_data`: Optional evaluation data
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate

**Returns:** Dictionary with training metrics

**Example:**
```python
train_data = [
    {"text": "Happy news!", "label": 1},
    {"text": "Sad story", "label": 0}
]

metrics = api.train_task(
    task_id="sentiment",
    train_data=train_data,
    epochs=5,
    batch_size=8,
    learning_rate=1e-5
)

print(f"Final accuracy: {metrics['train_accuracy']:.4f}")
```

##### predict()

```python
def predict(
    text: Union[str, List[str]], 
    task_id: str
) -> Dict[str, Any]
```

Make predictions on text input.

**Parameters:**
- `text`: Input text or list of texts
- `task_id`: Task to use for prediction

**Returns:** Dictionary with predictions and probabilities

**Example:**
```python
# Single text
result = api.predict("Great product!", "sentiment")

# Multiple texts
results = api.predict([
    "Excellent service",
    "Poor quality",
    "Average product"
], "sentiment")
```

##### evaluate_task()

```python
def evaluate_task(
    task_id: str, 
    eval_data: List[Dict[str, Any]]
) -> Dict[str, float]
```

Evaluate model performance on a task.

**Parameters:**
- `task_id`: Task identifier
- `eval_data`: Evaluation data

**Returns:** Evaluation metrics (accuracy, loss, etc.)

##### evaluate_all_tasks()

```python
def evaluate_all_tasks(
    eval_data_dict: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, float]]
```

Evaluate all trained tasks and compute continual learning metrics.

**Parameters:**
- `eval_data_dict`: Dictionary mapping task_id to evaluation data

**Returns:** Nested dictionary with metrics per task and overall CL metrics

##### get_task_info()

```python
def get_task_info() -> Dict[str, Any]
```

Get information about registered and trained tasks.

**Returns:**
```python
{
    "registered_tasks": ["sentiment", "topic"],
    "trained_tasks": ["sentiment"],
    "num_tasks": 2,
    "max_tasks": 50,
    "memory_usage": {...}
}
```

##### save() / load()

```python
def save(save_path: str) -> None

@classmethod
def load(cls, load_path: str, **kwargs) -> ContinualLearningAPI
```

Save and load trained models.

**Example:**
```python
# Save model
api.save("./my_model")

# Load model
loaded_api = ContinualLearningAPI.load("./my_model")
```

## 🚀 Deployment API

### ModelDeployment

Production deployment utilities.

#### Constructor

```python
from continual_transformer.deployment import ModelDeployment

deployment = ModelDeployment(api)
```

#### Methods

##### prepare_for_production()

```python
def prepare_for_production(
    optimization_level: str = "balanced",
    enable_monitoring: bool = True,
    max_memory_mb: Optional[int] = None
) -> Dict[str, Any]
```

Prepare model for production deployment.

**Parameters:**
- `optimization_level`: "speed", "memory", or "balanced"
- `enable_monitoring`: Enable runtime monitoring
- `max_memory_mb`: Memory limit in MB

##### health_check()

```python
def health_check() -> Dict[str, Any]
```

Run comprehensive health check.

**Returns:**
```python
{
    "status": "healthy",  # "healthy", "degraded", "unhealthy"
    "checks": {
        "model_loaded": {"passed": True, "message": "OK"},
        "memory_usage": {"passed": True, "message": "Normal"}
    },
    "warnings": [],
    "errors": []
}
```

##### benchmark_deployment()

```python
def benchmark_deployment(
    sample_texts: List[str],
    task_id: str,
    num_runs: int = 100
) -> Dict[str, Any]
```

Benchmark deployment performance.

**Returns:**
```python
{
    "average_inference_time_ms": 45.2,
    "throughput_samples_per_sec": 22.1,
    "success_rate": 0.99
}
```

##### export_deployment_package()

```python
def export_deployment_package(
    output_dir: str,
    include_examples: bool = True,
    include_docs: bool = True
) -> str
```

Export complete deployment package.

### BatchInferenceEngine

High-throughput async inference.

#### Constructor

```python
from continual_transformer.deployment import BatchInferenceEngine

engine = BatchInferenceEngine(
    model=api,
    max_batch_size=32,
    max_workers=4
)
```

#### Methods

##### predict_batch()

```python
def predict_batch(
    texts: List[str], 
    task_id: str,
    return_probabilities: bool = True
) -> List[Dict[str, Any]]
```

Efficient batch prediction.

**Example:**
```python
texts = ["Great!", "Terrible", "Okay", "Amazing!"]
results = engine.predict_batch(texts, "sentiment")

for result in results:
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {max(result['probabilities']):.3f}")
```

## 🔐 Security API

### SecurityValidator

Comprehensive input validation and security.

#### Constructor

```python
from continual_transformer.security.validator import SecurityValidator

validator = SecurityValidator()
```

#### Methods

##### validate_inference_request()

```python
def validate_inference_request(
    text: Union[str, List[str]],
    task_id: str,
    model_state: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]
```

Validate inference requests for security issues.

**Returns:** (is_valid, validation_report)

**Example:**
```python
is_valid, report = validator.validate_inference_request(
    text="Normal input text",
    task_id="sentiment"
)

if not is_valid:
    print(f"Validation failed: {report['errors']}")
```

### DataSanitizer

Input sanitization and cleaning.

```python
from continual_transformer.security.sanitizer import DataSanitizer

sanitizer = DataSanitizer()
clean_text = sanitizer.sanitize_text("<script>alert('xss')</script>Safe text")
print(clean_text)  # "Safe text"
```

## ⚡ Scaling API

### ScalingManager

Distributed computing and scaling utilities.

#### Constructor

```python
from continual_transformer.scaling import ScalingManager

scaling = ScalingManager(model, config={
    "max_batch_size": 32,
    "max_workers": 8
})
```

#### Methods

##### setup_async_inference()

```python
def setup_async_inference(**kwargs) -> None
```

Setup high-throughput async inference.

##### setup_load_balancing()

```python
def setup_load_balancing(model_instances: List[Any]) -> None
```

Setup load balancing across model instances.

##### get_scaling_status()

```python
def get_scaling_status() -> Dict[str, Any]
```

Get comprehensive scaling status.

## 🎯 Optimization API

### AutoTrainingLoop

Automatic hyperparameter optimization.

#### Constructor

```python
from continual_transformer.optimization.auto_optimization import AutoTrainingLoop

trainer = AutoTrainingLoop(
    model,
    enable_hyperparameter_optimization=True
)
```

#### Methods

##### auto_train()

```python
def auto_train(
    train_dataloader,
    eval_dataloader=None,
    num_epochs: int = 10,
    task_id: str = "auto_task",
    max_optimization_trials: int = 5
) -> Dict[str, Any]
```

Automatic training with hyperparameter optimization.

**Returns:**
```python
{
    "best_hyperparams": {"lr": 2e-5, "weight_decay": 0.01},
    "best_metrics": OptimizationMetrics(...),
    "optimization_history": [...]
}
```

## 📊 Configuration Classes

### ContinualConfig

Core configuration for continual learning.

```python
from continual_transformer.core import ContinualConfig

config = ContinualConfig(
    model_name="distilbert-base-uncased",
    max_tasks=50,
    device="cuda",
    learning_rate=2e-5,
    batch_size=16,
    max_sequence_length=512,
    # ... many more options
)
```

**Key Parameters:**
- `model_name`: Hugging Face model name
- `max_tasks`: Maximum number of tasks
- `device`: Computing device
- `learning_rate`: Default learning rate
- `batch_size`: Default batch size
- `adaptation_method`: Adaptation strategy
- `use_knowledge_distillation`: Enable knowledge distillation
- `elastic_weight_consolidation`: Enable EWC
- `freeze_base_model`: Freeze base model parameters

## 📈 Metrics and Monitoring

### ContinualMetrics

Continual learning specific metrics.

```python
from continual_transformer.metrics import ContinualMetrics

metrics = ContinualMetrics()

# Compute continual learning metrics
cl_metrics = metrics.compute_continual_metrics({
    "task1": {"accuracy": 0.85, "loss": 0.3},
    "task2": {"accuracy": 0.82, "loss": 0.4}
})

print(f"Average accuracy: {cl_metrics['average_accuracy']:.3f}")
print(f"Forgetting rate: {cl_metrics['forgetting_rate']:.3f}")
```

## 🔄 Error Handling

### Custom Exceptions

```python
from continual_transformer.core.exceptions import (
    TaskNotFoundError,
    ModelNotTrainedError,
    ValidationError,
    SecurityError
)

try:
    api.predict("text", "nonexistent_task")
except TaskNotFoundError as e:
    print(f"Task error: {e}")

try:
    validator.validate_inference_request("", "")
except ValidationError as e:
    print(f"Validation error: {e}")
```

## 🎮 Advanced Usage Examples

### Multi-Task Learning Pipeline

```python
# Initialize API
api = ContinualLearningAPI(max_tasks=10)

# Define tasks
tasks = [
    ("sentiment", 2, sentiment_data),
    ("emotion", 6, emotion_data),
    ("topic", 5, topic_data)
]

# Train tasks sequentially
for task_id, num_labels, data in tasks:
    print(f"Training {task_id}...")
    
    # Add task
    api.add_task(task_id, num_labels)
    
    # Train with optimization
    metrics = api.train_task(
        task_id=task_id,
        train_data=data["train"],
        eval_data=data["eval"],
        epochs=10
    )
    
    print(f"  Accuracy: {metrics['train_accuracy']:.3f}")

# Evaluate all tasks
eval_results = api.evaluate_all_tasks({
    task_id: data["test"] for task_id, _, data in tasks
})

print("Final Results:")
for task_id, metrics in eval_results.items():
    print(f"  {task_id}: {metrics['accuracy']:.3f}")
```

### Production Deployment

```python
# Setup for production
api = ContinualLearningAPI(device="cuda")

# Add and train tasks
api.add_task("sentiment", 2)
api.train_task("sentiment", train_data)

# Prepare deployment
from continual_transformer.deployment import ModelDeployment

deployment = ModelDeployment(api)

# Optimize for production
deployment_info = deployment.prepare_for_production(
    optimization_level="speed",
    enable_monitoring=True
)

# Run health checks
health = deployment.health_check()
if health["status"] != "healthy":
    print(f"Health issues: {health['warnings'] + health['errors']}")

# Benchmark performance
benchmark = deployment.benchmark_deployment(
    sample_texts=["Test text"] * 10,
    task_id="sentiment",
    num_runs=100
)

print(f"Average latency: {benchmark['average_inference_time_ms']:.1f}ms")
print(f"Throughput: {benchmark['throughput_samples_per_sec']:.1f} req/sec")

# Export deployment package
package_path = deployment.export_deployment_package("./production")
print(f"Deployment package: {package_path}")
```

### Async High-Throughput Inference

```python
import asyncio
from continual_transformer.deployment import AsyncInferenceEngine

async def main():
    # Setup async engine
    engine = AsyncInferenceEngine(
        model=api,
        max_batch_size=16,
        max_workers=4
    )
    
    await engine.start()
    
    try:
        # Submit multiple requests
        tasks = []
        for i in range(100):
            task = engine.predict_async({
                "text": f"Test text {i}",
                "task_id": "sentiment"
            })
            tasks.append(task)
        
        # Wait for all results
        results = await asyncio.gather(*tasks)
        
        # Process results
        accuracies = [max(r['probabilities'][0]) for r in results]
        avg_confidence = sum(accuracies) / len(accuracies)
        
        print(f"Processed {len(results)} requests")
        print(f"Average confidence: {avg_confidence:.3f}")
        
    finally:
        await engine.stop()

# Run async inference
asyncio.run(main())
```

## 📞 Support and Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   api.config.batch_size = 8
   
   # Or use CPU
   api = ContinualLearningAPI(device="cpu")
   ```

2. **Task Not Found**
   ```python
   # Check registered tasks
   info = api.get_task_info()
   print("Available tasks:", info["registered_tasks"])
   ```

3. **Validation Errors**
   ```python
   from continual_transformer.security.validator import SecurityValidator
   
   validator = SecurityValidator()
   is_valid, report = validator.validate_inference_request(text, task_id)
   
   if not is_valid:
       print("Validation issues:", report["errors"])
   ```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# API with debug info
api = ContinualLearningAPI(model_name="distilbert-base-uncased")
api.config.debug = True
```

### Performance Profiling

```python
# Enable performance monitoring
api.config.enable_profiling = True

# Get performance stats
stats = api.get_performance_stats()
print(f"Memory usage: {stats['memory_mb']:.1f}MB")
print(f"Inference time: {stats['avg_inference_ms']:.1f}ms")
```

For additional support, check:
- [GitHub Issues](https://github.com/your-org/continual-tiny-transformer/issues)
- [Documentation](https://continual-tiny-transformer.readthedocs.io/)
- [Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)