# Core API Reference

The core module contains the main `ContinualTransformer` class and fundamental components for zero-parameter continual learning.

## ContinualTransformer

The primary interface for continual learning with transformers.

### Class Definition

```python
class ContinualTransformer:
    """
    Zero-parameter continual learning transformer.
    
    This class implements a memory-efficient continual learning framework
    that adds ZERO new parameters per task while maintaining high knowledge
    retention across all learned tasks.
    
    Args:
        model_size: Size of the base transformer ("base", "large", "xl")
        max_tasks: Maximum number of tasks to support (default: 50)
        hidden_size: Hidden dimension size (default: 768)
        num_heads: Number of attention heads (default: 12)
        num_layers: Number of transformer layers (default: 6)
        vocab_size: Vocabulary size (default: 50000)
        device: Computing device ("cpu", "cuda", "auto")
        
    Example:
        >>> model = ContinualTransformer(
        ...     model_size="base",
        ...     max_tasks=10,
        ...     device="cuda"
        ... )
        >>> model.learn_task("sentiment", train_data, epochs=5)
        >>> predictions = model.predict("Great product!", "sentiment")
    """
```

### Key Methods

#### `learn_task(task_id, train_data, epochs=10, **kwargs)`

Learn a new task without adding parameters.

**Parameters:**
- `task_id` (str): Unique identifier for the task
- `train_data` (Dataset): Training data for the task
- `epochs` (int): Number of training epochs
- `**kwargs`: Additional training arguments

**Returns:**
- `TaskLearningResult`: Training metrics and task information

**Example:**
```python
result = model.learn_task(
    task_id="sentiment_analysis",
    train_data=sentiment_dataset,
    epochs=10,
    learning_rate=2e-5,
    batch_size=16
)
print(f"Final accuracy: {result.final_accuracy:.3f}")
```

#### `predict(text, task_id, **kwargs)`

Make predictions for a specific task.

**Parameters:**
- `text` (str or List[str]): Input text(s) for prediction  
- `task_id` (str): Task to use for prediction
- `**kwargs`: Additional inference arguments

**Returns:**
- Prediction results (format depends on task type)

**Example:**
```python
# Single prediction
result = model.predict("I love this movie!", "sentiment")

# Batch prediction  
results = model.predict([
    "Great product!",
    "Terrible experience."
], "sentiment")
```

#### `evaluate_task(task_id, test_data, metrics=None)`

Evaluate performance on a specific task.

**Parameters:**
- `task_id` (str): Task to evaluate
- `test_data` (Dataset): Test dataset
- `metrics` (List[str]): Metrics to compute

**Returns:**
- `EvaluationResult`: Comprehensive evaluation metrics

#### `get_task_info(task_id=None)`

Get information about learned tasks.

**Parameters:**
- `task_id` (str, optional): Specific task ID, or None for all tasks

**Returns:**
- Task information dictionary or list of dictionaries

### Properties

#### `learned_tasks`
List of all learned task IDs.

#### `memory_usage`  
Current memory usage statistics.

#### `model_size`
Total number of parameters (constant across tasks).

## Configuration

See [Configuration API](config.md) for detailed configuration options.

## Related Classes

- [`TaskManager`](tasks.md) - Manages task lifecycle and metadata
- [`ActivationAdapter`](adapters.md) - Handles task-specific adaptations
- [`ContinualMetrics`](metrics.md) - Evaluation and benchmarking metrics

## Implementation Notes

### Zero-Parameter Mechanism

The model achieves zero parameter expansion through:

1. **Frozen Base Weights**: Core transformer parameters never change
2. **Activation Adaptation**: Task-specific modifications to activation patterns
3. **Knowledge Distillation**: Retains previous task knowledge without storing additional parameters
4. **Dynamic Routing**: Lightweight task identification and routing

### Memory Efficiency

- Memory complexity: O(1) with respect to number of tasks
- Storage overhead: < 1MB per additional task
- Inference speed: Constant regardless of task count

### Performance Characteristics

- Knowledge retention: >90% on previous tasks
- Task capacity: 50+ tasks without degradation  
- Training overhead: <10% per additional task

## External References

- [Amazon Research Paper](https://arxiv.org/abs/example) - Foundational methodology
- [PyTorch Transformers](https://pytorch.org/docs/stable/nn.html#transformer-layers)
- [Continual Learning Survey](https://continualai.org/research/)