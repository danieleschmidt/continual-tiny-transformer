# Quick Start Guide

This guide will get you up and running with Continual Tiny Transformer in just a few minutes.

## Installation

```bash
# Install from PyPI (recommended)
pip install continual-tiny-transformer

# Or install from source
git clone https://github.com/your-org/continual-tiny-transformer.git
cd continual-tiny-transformer
pip install -e .
```

## Basic Usage

### 1. Initialize the Model

```python
from continual_transformer import ContinualTransformer

# Create a model that can learn up to 50 tasks
model = ContinualTransformer(
    model_size="base",    # Options: "base", "large", "xl"
    max_tasks=50,         # Maximum number of tasks
    device="auto"         # Auto-detect GPU/CPU
)

print(f"Model initialized with {model.model_size:,} parameters")
# Output: Model initialized with 110,000,000 parameters
```

### 2. Prepare Your Data

```python
# Example: Sentiment analysis data
sentiment_data = [
    {"text": "I love this product!", "label": 1},
    {"text": "This is terrible.", "label": 0},
    {"text": "Amazing quality!", "label": 1},
    {"text": "Waste of money.", "label": 0},
]

# Example: Text classification data  
topic_data = [
    {"text": "The stock market rose today.", "label": "finance"},
    {"text": "New AI breakthrough announced.", "label": "technology"},
    {"text": "Team wins championship.", "label": "sports"},
    {"text": "Election results announced.", "label": "politics"},
]
```

### 3. Learn Your First Task

```python
# Learn sentiment analysis (Task 1)
result = model.learn_task(
    task_id="sentiment",
    train_data=sentiment_data,
    epochs=10,
    learning_rate=2e-5,
    batch_size=16
)

print(f"Task learned! Final accuracy: {result.final_accuracy:.3f}")
print(f"Parameters added: {result.parameters_added}")  # Always 0!
```

### 4. Learn Additional Tasks (Zero New Parameters!)

```python
# Learn topic classification (Task 2) 
# Note: No new parameters are added!
result = model.learn_task(
    task_id="topic",
    train_data=topic_data,
    epochs=10
)

print(f"Total tasks learned: {len(model.learned_tasks)}")
print(f"Model parameters: {model.model_size}")  # Still the same!
```

### 5. Make Predictions

```python
# Predict with the sentiment task
sentiment_result = model.predict(
    text="This is an excellent product!",
    task_id="sentiment"
)
print(f"Sentiment: {'Positive' if sentiment_result > 0.5 else 'Negative'}")

# Predict with the topic task
topic_result = model.predict(
    text="Scientists discover new exoplanet.",
    task_id="topic"
)
print(f"Topic: {topic_result}")

# Batch predictions
texts = [
    "I hate this movie.",
    "This film is amazing!",
    "Okay movie, nothing special."
]
batch_results = model.predict(texts, task_id="sentiment")
```

### 6. Evaluate Performance

```python
# Prepare test data
test_data = [
    {"text": "Great service!", "label": 1},
    {"text": "Poor quality.", "label": 0},
]

# Evaluate the sentiment task
eval_result = model.evaluate_task(
    task_id="sentiment",
    test_data=test_data,
    metrics=["accuracy", "precision", "recall", "f1"]
)

print(f"Test accuracy: {eval_result.accuracy:.3f}")
print(f"F1 score: {eval_result.f1:.3f}")
```

## Key Features Demonstrated

### ✅ Zero Parameter Expansion
```python
# Before learning any tasks
initial_params = model.model_size

# After learning 5 tasks
for i in range(5):
    model.learn_task(f"task_{i}", task_data, epochs=5)

final_params = model.model_size
print(f"Parameter growth: {final_params - initial_params}")  # 0!
```

### ✅ Knowledge Retention
```python
# Test retention on first task after learning more tasks
model.learn_task("task_1", data_1, epochs=10)
accuracy_1 = model.evaluate_task("task_1", test_1).accuracy

model.learn_task("task_2", data_2, epochs=10)  
model.learn_task("task_3", data_3, epochs=10)

# Task 1 accuracy should still be >90% of original
retention_accuracy = model.evaluate_task("task_1", test_1).accuracy
retention_rate = retention_accuracy / accuracy_1

print(f"Knowledge retention: {retention_rate:.1%}")  # >90%
```

### ✅ Memory Efficiency
```python
# Memory usage remains constant
memory_before = model.memory_usage
model.learn_task("new_task", new_data, epochs=10)
memory_after = model.memory_usage

print(f"Memory increase: {memory_after - memory_before} MB")  # ~0 MB
```

## What's Next?

- **[Installation Guide](02-installation.md)** - Detailed installation options
- **[Configuration](03-configuration.md)** - Customize your model
- **[Understanding Continual Learning](04-continual-learning.md)** - Learn the concepts
- **[Examples](examples/)** - More detailed examples

## Common Issues

### GPU Memory
```python
# If you encounter GPU memory issues
model = ContinualTransformer(
    model_size="base",    # Try smaller model
    device="cpu",         # Use CPU instead
    max_tasks=10          # Reduce task limit
)
```

### Import Errors  
```bash
# Make sure you have the right dependencies
pip install torch transformers datasets
```

### Performance
```python
# For better performance
model = ContinualTransformer(
    model_size="large",   # Larger models perform better
    device="cuda",        # Use GPU if available
    precision="fp16"      # Use half precision
)
```

## Getting Help

- 📖 [Full Documentation](../api/)
- 🐛 [Report Issues](https://github.com/your-org/continual-tiny-transformer/issues)  
- 💬 [Discussions](https://github.com/your-org/continual-tiny-transformer/discussions)
- 📧 [Contact](mailto:daniel@example.com)