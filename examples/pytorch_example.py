"""
PyTorch example: train a small MLP on a synthetic dataset, tracking per-epoch metrics.

Requires: torch
    pip install torch --index-url https://download.pytorch.org/whl/cpu

Run from the repo root:
    python examples/pytorch_example.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch is not installed. Install it with:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

from tracker import Experiment


# ── Synthetic dataset ──────────────────────────────────────────────
torch.manual_seed(0)
X = torch.randn(500, 10)
y = (X[:, 0] + X[:, 1] > 0).float()
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# ── Simple MLP ─────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Train with two different hidden sizes ──────────────────────────
for hidden in [16, 64]:
    with Experiment(
        name=f"mlp-hidden{hidden}",
        tags={"framework": "pytorch", "task": "binary_classification"},
    ) as exp:
        exp.log_params({"hidden_size": hidden, "lr": 1e-3, "epochs": 10, "batch_size": 32})

        model = MLP(hidden=hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        for epoch in range(10):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
                correct += ((preds > 0.5) == yb.bool()).sum().item()
                total += len(xb)

            avg_loss = total_loss / total
            accuracy = correct / total
            exp.log_metrics({"loss": avg_loss, "accuracy": accuracy}, step=epoch)
            print(f"  Epoch {epoch+1:02d}  loss={avg_loss:.4f}  acc={accuracy:.4f}")

print("\nDone! Run `python -m tracker compare --metric accuracy` to compare runs.")
