import json
import shutil
from pathlib import Path


class Storage:
    """Handles reading and writing run data to the local filesystem."""

    def __init__(self, base_dir: str, run_id: str):
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, payload: dict):
        """Persist a run's metadata to meta.json."""
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(payload, f, indent=2)

    def save_artifact(self, filepath: str):
        """Copy an artifact file into the run's artifacts/ subfolder."""
        dst = self.run_dir / "artifacts"
        dst.mkdir(exist_ok=True)
        shutil.copy(filepath, dst)
        return str(dst / Path(filepath).name)

    @staticmethod
    def load_all_runs(base_dir: str = "runs") -> list:
        """Load all completed runs from the base directory."""
        runs = []
        for meta_file in sorted(Path(base_dir).glob("*/meta.json")):
            try:
                with open(meta_file) as f:
                    runs.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        return runs

    @staticmethod
    def load_run(run_id: str, base_dir: str = "runs") -> dict:
        """Load a single run by run_id."""
        meta_file = Path(base_dir) / run_id / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"No run found with id '{run_id}' in '{base_dir}'")
        with open(meta_file) as f:
            return json.load(f)

    @staticmethod
    def delete_run(run_id: str, base_dir: str = "runs"):
        """Delete a run directory entirely."""
        run_dir = Path(base_dir) / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"No run found with id '{run_id}'")
        shutil.rmtree(run_dir)
