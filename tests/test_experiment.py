"""
Unit tests for the experiment tracker.

Run with:
    python -m pytest tests/ -v
"""
import json
import os
import shutil
import tempfile
import time
import pytest

from tracker.experiment import Experiment
from tracker.storage import Storage


@pytest.fixture
def tmp_dir():
    """Create a temp directory for each test and clean up after."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ------------------------------------------------------------------ #
#  Experiment basics                                                   #
# ------------------------------------------------------------------ #

class TestExperimentCore:
    def test_run_creates_meta_file(self, tmp_dir):
        exp = Experiment("test-run", base_dir=tmp_dir)
        exp.end_run()
        meta_files = list(__import__("pathlib").Path(tmp_dir).glob("*/meta.json"))
        assert len(meta_files) == 1

    def test_run_id_is_unique(self, tmp_dir):
        exp1 = Experiment("run-a", base_dir=tmp_dir)
        exp1.end_run()
        exp2 = Experiment("run-b", base_dir=tmp_dir)
        exp2.end_run()
        assert exp1.run_id != exp2.run_id

    def test_log_params(self, tmp_dir):
        exp = Experiment("param-test", base_dir=tmp_dir)
        exp.log_params({"lr": 0.01, "batch_size": 32})
        exp.end_run()
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["params"]["lr"] == 0.01
        assert run["params"]["batch_size"] == 32

    def test_log_param_single(self, tmp_dir):
        exp = Experiment("single-param", base_dir=tmp_dir)
        exp.log_param("dropout", 0.5)
        exp.end_run()
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["params"]["dropout"] == 0.5

    def test_log_metrics_builds_history(self, tmp_dir):
        exp = Experiment("metric-test", base_dir=tmp_dir)
        for i in range(5):
            exp.log_metrics({"loss": 1.0 - i * 0.1}, step=i)
        exp.end_run()
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert len(run["metrics"]["loss"]) == 5
        assert run["metrics"]["loss"][0]["value"] == pytest.approx(1.0)
        assert run["metrics"]["loss"][-1]["value"] == pytest.approx(0.6)

    def test_log_metric_step_recorded(self, tmp_dir):
        exp = Experiment("step-test", base_dir=tmp_dir)
        exp.log_metric("accuracy", 0.9, step=3)
        exp.end_run()
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["metrics"]["accuracy"][0]["step"] == 3

    def test_status_completed(self, tmp_dir):
        exp = Experiment("status-test", base_dir=tmp_dir)
        exp.end_run()
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["status"] == "completed"

    def test_status_failed(self, tmp_dir):
        exp = Experiment("fail-test", base_dir=tmp_dir)
        exp.fail_run(reason="OOM")
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["status"] == "failed"

    def test_notes(self, tmp_dir):
        exp = Experiment("note-test", base_dir=tmp_dir)
        exp.set_notes("This run uses a new scheduler.")
        exp.end_run()
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert "new scheduler" in run["notes"]

    def test_duration_is_recorded(self, tmp_dir):
        exp = Experiment("duration-test", base_dir=tmp_dir)
        time.sleep(0.05)
        exp.end_run()
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["duration_s"] >= 0.05

    def test_tags_stored(self, tmp_dir):
        exp = Experiment("tag-test", tags={"dataset": "iris", "v": "2"}, base_dir=tmp_dir)
        exp.end_run()
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["tags"]["dataset"] == "iris"


# ------------------------------------------------------------------ #
#  Context manager                                                     #
# ------------------------------------------------------------------ #

class TestContextManager:
    def test_context_manager_completes(self, tmp_dir):
        with Experiment("ctx-test", base_dir=tmp_dir) as exp:
            exp.log_param("x", 1)
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["status"] == "completed"

    def test_context_manager_fails_on_exception(self, tmp_dir):
        with pytest.raises(ValueError):
            with Experiment("ctx-fail", base_dir=tmp_dir) as exp:
                raise ValueError("something went wrong")
        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert run["status"] == "failed"


# ------------------------------------------------------------------ #
#  Artifact logging                                                    #
# ------------------------------------------------------------------ #

class TestArtifacts:
    def test_artifact_copied(self, tmp_dir):
        # Create a dummy file to act as an artifact
        artifact = os.path.join(tmp_dir, "model.txt")
        with open(artifact, "w") as f:
            f.write("weights")

        exp = Experiment("artifact-test", base_dir=tmp_dir)
        exp.log_artifact(artifact)
        exp.end_run()

        run = Storage.load_run(exp.run_id, base_dir=tmp_dir)
        assert len(run["artifacts"]) == 1
        assert os.path.exists(run["artifacts"][0])


# ------------------------------------------------------------------ #
#  Storage                                                             #
# ------------------------------------------------------------------ #

class TestStorage:
    def test_load_all_runs(self, tmp_dir):
        for name in ["run-a", "run-b", "run-c"]:
            exp = Experiment(name, base_dir=tmp_dir)
            exp.end_run()
        runs = Storage.load_all_runs(base_dir=tmp_dir)
        assert len(runs) == 3

    def test_load_run_not_found(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            Storage.load_run("nonexistent", base_dir=tmp_dir)

    def test_delete_run(self, tmp_dir):
        exp = Experiment("delete-me", base_dir=tmp_dir)
        exp.end_run()
        Storage.delete_run(exp.run_id, base_dir=tmp_dir)
        with pytest.raises(FileNotFoundError):
            Storage.load_run(exp.run_id, base_dir=tmp_dir)
