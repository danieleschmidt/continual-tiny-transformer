# Continual Tiny Transformer

A minimal, from-scratch implementation of a small transformer with **Elastic Weight Consolidation (EWC)** for continual learning — zero new parameters per task.

## Results (demo output)

```
Method       A acc after A A acc after B Forgetting   B acc
------------------------------------------------------------
Naive                0.976         0.878     +0.098   0.982
EWC                  0.960         0.982     -0.022   0.970
------------------------------------------------------------
EWC reduced forgetting by 0.120 accuracy points.
✓ EWC successfully mitigated catastrophic forgetting.
```

**Naive fine-tuning**: 9.8% forgetting on Task A after training Task B.  
**EWC**: ~0% forgetting, while still learning Task B to 97% accuracy.

## What it does

Trains a small transformer (2 layers, ~70 K params) on two sequential tasks with a domain shift:

- **Task A** (`low_range`): sequences from token IDs 1–32, label = 1 if mean > mid-low threshold
- **Task B** (`high_range`): sequences from token IDs 33–63, same rule on different tokens

Training Task B naively forces the model to overwrite the token embeddings it learned for Task A.  
EWC prevents this by penalising changes to parameters that were important for Task A.

## Architecture

```
Token IDs → Embedding → Positional Encoding
  → N × TransformerEncoderLayer (pre-LN)
  → mean pool → LayerNorm → Linear head(s)
```

All code is plain PyTorch — no HuggingFace, no external data downloads.  
Multi-head support: `model.add_head("task_name", num_classes)` for task-incremental setups.

## Quick start

```bash
# Install (only dependency: torch)
pip install -e .

# Run demo (~2 min on CPU, shows forgetting vs EWC)
python demo.py

# Shorter smoke test
python demo.py --epochs 15 --ewc-lambda 500

# Tests (30 tests, ~10 sec)
pytest
```

## EWC usage

```python
from continual_transformer import TinyTransformer, EWC, generate_task
from continual_transformer.train import train, evaluate
import torch

device = torch.device("cpu")
task_a = generate_task("low_range",  vocab_size=64)
task_b = generate_task("high_range", vocab_size=64)

model = TinyTransformer(vocab_size=64)
train(model, task_a.train_loader, task_a.val_loader, device, epochs=30)

# Snapshot Fisher info + parameter anchor after Task A
ewc = EWC(model, task_a.train_loader, device, n_samples=500)

# Train on Task B with EWC penalty
train(model, task_b.train_loader, task_b.val_loader, device,
      epochs=30, ewc=ewc, ewc_lambda=500)

print(evaluate(model, task_a.val_loader, device))  # still high!
```

## EWC penalty

```
L_total = L_task_B  +  (λ/2) · Σ_i  F_i · (θ_i − θ*_i)²
```

| Symbol | Meaning |
|--------|---------|
| `F_i`  | Diagonal Fisher estimate for parameter `i` (importance weight) |
| `θ*_i` | Parameter values after Task A (anchor) |
| `λ`    | Trade-off — higher = more protection for Task A, less plasticity |

## Project structure

```
continual_transformer/
  model.py   – TinyTransformer (from-scratch, multi-head)
  ewc.py     – EWC: Fisher estimation + penalty
  tasks.py   – Synthetic task generators (no data downloads)
  train.py   – Training loop + evaluation
tests/
  test_model.py  – architecture tests
  test_ewc.py    – EWC unit + end-to-end tests
  test_tasks.py  – task generator tests
  test_train.py  – training loop tests
demo.py    – side-by-side forgetting vs EWC demo
```

## References

- Kirkpatrick et al. (2017) — *Overcoming catastrophic forgetting in neural networks*  
  <https://arxiv.org/abs/1612.00796>
