import uuid
import time
from typing import Optional
from .storage import Storage


class Experiment:
    """
    Tracks a single model training run — params, metrics, and artifacts.

    Usage:
        exp = Experiment(name="my-run", tags={"dataset": "iris"})
        exp.log_params({"lr": 0.01, "epochs": 10})
        for epoch in range(10):
            exp.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=epoch)
        exp.end_run()
    """

    def __init__(
        self,
        name: str,
        tags: Optional[dict] = None,
        base_dir: str = "runs",
        run_id: Optional[str] = None,
    ):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.name = name
        self.tags = tags or {}
        self.params: dict = {}
        self.metrics: dict = {}
        self.artifacts: list = []
        self.notes: str = ""
        self.start_time = time.time()
        self.status = "running"
        self.storage = Storage(base_dir, self.run_id)
        self._base_dir = base_dir

        # Save an in-progress snapshot immediately
        self._save(status="running")
        print(f"🚀 Run started  →  name: '{self.name}'  |  run_id: {self.run_id}")

    # ------------------------------------------------------------------ #
    #  Logging API                                                         #
    # ------------------------------------------------------------------ #

    def log_params(self, params: dict):
        """Log a dict of hyperparameters (overrides duplicates)."""
        self.params.update(params)
        self._save()

    def log_param(self, key: str, value):
        """Log a single hyperparameter."""
        self.log_params({key: value})

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log a dict of scalar metrics.  Call multiple times to build a history.

        Args:
            metrics: e.g. {"loss": 0.42, "val_accuracy": 0.91}
            step:    optional epoch / iteration number
        """
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append({"step": step, "value": float(v)})
        self._save()

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single scalar metric."""
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, filepath: str):
        """Copy a file (model weights, plots, CSVs) into the run folder."""
        saved_path = self.storage.save_artifact(filepath)
        self.artifacts.append(saved_path)
        self._save()
        print(f"📎 Artifact saved → {saved_path}")

    def set_notes(self, text: str):
        """Add free-form notes to the run."""
        self.notes = text
        self._save()

    # ------------------------------------------------------------------ #
    #  Finalisation                                                        #
    # ------------------------------------------------------------------ #

    def end_run(self) -> str:
        """Finalise the run and persist everything. Returns run_id."""
        self.status = "completed"
        self._save(status="completed")
        duration = round(time.time() - self.start_time, 2)
        print(f"✅ Run complete  →  run_id: {self.run_id}  |  duration: {duration}s")
        return self.run_id

    def fail_run(self, reason: str = ""):
        """Mark the run as failed."""
        self.status = "failed"
        if reason:
            self.notes = f"[FAILED] {reason}\n" + self.notes
        self._save(status="failed")
        print(f"❌ Run failed    →  run_id: {self.run_id}  |  reason: {reason}")

    # ------------------------------------------------------------------ #
    #  Context manager support                                             #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.fail_run(reason=str(exc_val))
            return False  # re-raise the exception
        self.end_run()
        return False

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _to_dict(self, status: Optional[str] = None) -> dict:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "tags": self.tags,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "duration_s": round(time.time() - self.start_time, 2),
            "status": status or self.status,
            "start_time": self.start_time,
        }

    def _save(self, status: Optional[str] = None):
        self.storage.save_run(self._to_dict(status))
