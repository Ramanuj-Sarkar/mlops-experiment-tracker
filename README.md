# 🧪 MLOps Experiment Tracker

A lightweight, **zero-mandatory-dependency** experiment tracker for Python ML projects.

No server to spin up. No config files. Just Python.

```python
from tracker import Experiment

with Experiment("logreg-baseline", tags={"dataset": "iris"}) as exp:
    exp.log_params({"C": 0.1, "solver": "lbfgs"})
    for epoch in range(10):
        exp.log_metrics({"loss": 0.5 - epoch * 0.03, "val_accuracy": 0.8 + epoch * 0.01}, step=epoch)
```

---

## Why this instead of MLflow?

| Feature | mlops-experiment-tracker | MLflow |
|---|---|---|
| Zero mandatory deps | ✅ | ❌ (heavy) |
| No server required | ✅ | ❌ (tracking server) |
| Familiar API | ✅ (`log_params` / `log_metrics`) | ✅ |
| HTML reports | ✅ | ✅ |
| Model registry | ❌ | ✅ |
| UI dashboard | ❌ | ✅ |

**Use this when:** you want fast, local experiment tracking without infra overhead.  
**Use MLflow when:** you need a full UI, model registry, or team collaboration.

---

## Installation

```bash
git clone https://github.com/Ramanuj-Sarkar/mlops-experiment-tracker
cd mlops-experiment-tracker

# Core only (no deps needed)
pip install -e .

# With report generation
pip install -e ".[reports]"

# With sklearn examples
pip install -e ".[reports,examples]"
```

---

## Quick Start

### Log a run

```python
from tracker import Experiment

exp = Experiment(
    name="my-model-v1",
    tags={"dataset": "my_data", "author": "ramanuj"}
)

exp.log_params({"learning_rate": 0.001, "epochs": 20, "batch_size": 64})

for epoch in range(20):
    # ... your training loop ...
    exp.log_metrics({"train_loss": 0.5, "val_loss": 0.48, "val_accuracy": 0.91}, step=epoch)

exp.log_artifact("model_weights.pt")   # optional — copy a file into the run
exp.end_run()
```

### Context manager (auto end_run / fail_run)

```python
with Experiment("safe-run") as exp:
    exp.log_params({"lr": 0.001})
    # If an exception is raised here, the run is automatically marked "failed"
    train_model()
# end_run() is called automatically on success
```

---

## CLI

```bash
# List all runs
python -m tracker list

# Compare runs by a metric
python -m tracker compare --metric val_accuracy

# Show full JSON for one run
python -m tracker show <run_id>

# Generate an HTML report
python -m tracker report --metric val_accuracy --output report.html

# Delete failed runs
python -m tracker clean --status failed
```

---

## Project Structure

```
mlops-experiment-tracker/
├── tracker/
│   ├── __init__.py       # Public API
│   ├── experiment.py     # Experiment / run tracking
│   ├── storage.py        # Filesystem backend (pure stdlib)
│   ├── report.py         # HTML report + pandas comparison
│   └── __main__.py       # CLI
├── examples/
│   ├── sklearn_example.py
│   └── pytorch_example.py
├── tests/
│   └── test_experiment.py
├── requirements.txt
├── setup.py
└── README.md
```

Each run is stored as a folder under `runs/`:

```
runs/
└── a3f1b2c4/
    ├── meta.json          # params, metrics, tags, status
    └── artifacts/         # any files you log_artifact()'d
```

---

## Running the Examples

```bash
# Sklearn: compare Logistic Regression with different C values
python examples/sklearn_example.py

# PyTorch: train a small MLP (requires torch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
python examples/pytorch_example.py
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## License

MIT
