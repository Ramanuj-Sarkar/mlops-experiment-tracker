"""
Scikit-learn example: compare Logistic Regression runs with different C values.

Run from the repo root:
    python examples/sklearn_example.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tracker import Experiment, compare_runs

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for C in [0.01, 0.1, 1.0, 10.0]:
    with Experiment(
        name=f"logreg-C{C}",
        tags={"dataset": "iris", "model": "logistic_regression"},
    ) as exp:
        exp.log_params({"C": C, "solver": "lbfgs", "max_iter": 200})

        model = LogisticRegression(C=C, solver="lbfgs", max_iter=200)
        model.fit(X_train, y_train)

        # Simulate per-epoch metrics (LogReg trains in one shot, so we fake steps)
        for i, (train_score, test_score) in enumerate(
            zip([0.7, 0.8, 0.85, 0.9], [0.65, 0.75, 0.82, accuracy_score(y_test, model.predict(X_test))])
        ):
            exp.log_metrics({"train_accuracy": train_score, "val_accuracy": test_score}, step=i)

        f1 = f1_score(y_test, model.predict(X_test), average="weighted")
        exp.log_metric("f1_weighted", f1)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
df = compare_runs(metric="val_accuracy")
print(df[["run_id", "name", "C", "val_accuracy", "f1_weighted"]].to_string(index=False))
