# API Documentation

This directory contains comprehensive API documentation for the Continual Tiny Transformer library.

## Structure

- **[Core API](core.md)** - Main ContinualTransformer class and core functionality
- **[Configuration](config.md)** - Configuration classes and settings
- **[Tasks](tasks.md)** - Task management and lifecycle
- **[Adapters](adapters.md)** - Adaptation mechanisms for continual learning
- **[Metrics](metrics.md)** - Evaluation metrics and benchmarking
- **[Utilities](utils.md)** - Helper functions and utilities

## Quick Reference

### Basic Usage

```python
from continual_transformer import ContinualTransformer

# Initialize model
model = ContinualTransformer(
    model_size="base",
    max_tasks=50
)

# Learn first task
model.learn_task(
    task_id="sentiment",
    train_data=train_data,
    epochs=10
)

# Learn additional tasks (zero new parameters)
model.learn_task(
    task_id="summarization",
    train_data=summary_data, 
    epochs=10
)

# Inference
result = model.predict(
    text="Great product!",
    task_id="sentiment"
)
```

### Key Features

- **Zero Parameter Expansion**: Add new tasks without increasing model size
- **High Knowledge Retention**: >90% accuracy on previous tasks
- **Scalable Architecture**: Support for 50+ tasks
- **Memory Efficient**: Constant memory usage regardless of task count

## API Reference Generation

This documentation is automatically generated from docstrings using Sphinx. To rebuild:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Generate API docs
sphinx-apidoc -o docs/api/ src/continual_transformer/

# Build documentation
cd docs && make html
```

## Contributing to Documentation

When adding new functionality:

1. Include comprehensive docstrings following Google style
2. Add type hints for all parameters and return values
3. Include usage examples in docstrings
4. Update this index with new modules
5. Run documentation tests: `make docs`

## External References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Continual Learning Papers](https://continualai.org/research/)
- [Amazon Research Blog](https://www.amazon.science/blog)