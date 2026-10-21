import io
import json

import pytest

from video_automation import media
from video_automation.cli import main
from video_automation.models import Config, Cue, Manifest
from video_automation.pipeline import Pipeline
from video_automation.progress import WorkflowProgress
from video_automation.store import PipelineError, Store, TemporaryError, read_json, worker_lock


@pytest.fixture
def store(tmp_path):
    config = Config.load().model_copy(update={"jobs_dir": tmp_path / "jobs", "width": 640, "height": 360,
        "max_attempts": 2, "retry_backoff_seconds": 0, "image_interval_seconds": 0})
    manifest = Manifest(job_id="progress", topic="Progress fixture", mode="fixture", cues=[
        Cue(id="cue_0001", text="Scene.", start=0, end=1, word_start=0, word_end=1)])
    return Store(config.jobs_dir / manifest.job_id, config, manifest)


def snapshot_files(folder):
    return {path.relative_to(folder): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in folder.rglob("*") if path.is_file()}


def test_retries_and_cache_hits_have_correct_progress_counts(store):
    store.progress_stream = io.StringIO()
    progress = WorkflowProgress(store, ["images"])
    store.progress = progress
    calls = []

    def generate():
        calls.append(1)
        snapshot = read_json(store.path("progress.json"))
        assert snapshot["completed_units"] == 0
        assert snapshot["attempt"] == len(calls)
        if len(calls) == 1:
            raise TemporaryError("Retry fixture")
        return {"files": []}

    with progress.phase("images"):
        store.task("image.cue_0001", {}, generate)
        snapshot = read_json(store.path("progress.json"))
        assert snapshot["completed_units"] == snapshot["total_units"] == 1
        assert snapshot["task_status"] == "completed"
        store.task("image.cue_0001", {}, generate)
        assert read_json(store.path("progress.json"))["completed_units"] == 1
    progress.finish("completed")
    resumed = WorkflowProgress(store, ["images"])
    store.progress = resumed
    with resumed.phase("images"):
        store.task("image.cue_0001", {}, generate)
        snapshot = read_json(store.path("progress.json"))
        assert snapshot["cached_units"] == snapshot["completed_units"] == 1
        assert snapshot["task_status"] == "cached"
    assert len(calls) == 2
    output = store.progress_stream.getvalue()
    assert "attempt 1/2" in output and "attempt 2/2" in output
    assert "1/1 images" in output and "cached" in output


def test_exhausted_retry_reports_failed_task_without_new_work(store):
    calls = []

    def generate():
        calls.append(1)
        raise TemporaryError("Retry fixture")

    with pytest.raises(TemporaryError):
        store.task("image.cue_0001", {}, generate)
    progress = WorkflowProgress(store, ["images"])
    store.progress = progress
    with pytest.raises(PipelineError, match="retry limit reached"):
        with progress.phase("images"):
            store.task("image.cue_0001", {}, generate)
    snapshot = read_json(store.path("progress.json"))
    assert snapshot["task"] == "image.cue_0001" and snapshot["task_status"] == "failed"
    assert snapshot["attempt"] == snapshot["max_attempts"] == 2
    assert snapshot["completed_units"] == 0 and len(calls) == 2


@pytest.mark.parametrize("error,status", [(PipelineError("Render fixture failed"), "failed"),
                                        (KeyboardInterrupt(), "interrupted")])
def test_workflow_progress_preserves_failure_and_interruption_state(store, monkeypatch, error, status):
    def render(store):
        raise error
    monkeypatch.setattr(media, "render", render)
    with pytest.raises(type(error)):
        Pipeline(store).run()
    snapshot = read_json(store.path("progress.json"))
    assert snapshot["status"] == snapshot["phase_status"] == status
    assert snapshot["phase"] == "render"
    assert snapshot["completed_phases"] == snapshot["completed_units"] == 0
    assert snapshot["total_phases"] == 3
    assert snapshot["error"] == store.manifest.error
    assert store.manifest.status == status
    assert store.progress is None


@pytest.mark.parametrize("has_progress", [False, True])
@pytest.mark.parametrize("use_manifest_path", [False, True])
def test_status_is_read_only_and_available_while_worker_is_locked(store, tmp_path, capsys, has_progress, use_manifest_path):
    settings = tmp_path / "config.json"
    settings.write_text(store.config.model_dump_json(), encoding="utf-8")
    if has_progress:
        store.manifest.status = "running"
        store.save()
        progress = WorkflowProgress(store, ["render", "finish", "validate"])
        store.progress = progress
        with progress.phase("render"):
            before = snapshot_files(store.folder)
            with worker_lock(store.config.jobs_dir):
                target = store.path("manifest.json") if use_manifest_path else store.folder
                assert main(["--config", str(settings), "status", str(target)]) == 0
            report = json.loads(capsys.readouterr().out)
            assert report["status"] == "running" and report["phase"] == "render"
            assert report["completed_phases"] == 0 and report["total_phases"] == 3
            assert report["total_units"] == 1
            assert snapshot_files(store.folder) == before
    else:
        # Historical manifests remain readable without constructing a writing Store.
        manifest = store.manifest.model_dump(mode="json")
        manifest["mode"] = "production"
        manifest.pop("image_strategy")
        store.path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        before = snapshot_files(store.folder)
        with worker_lock(store.config.jobs_dir):
            target = store.path("manifest.json") if use_manifest_path else store.folder
            assert main(["--config", str(settings), "status", str(target)]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report == {"job_id": "progress", "mode": "production", "status": "pending",
                          "error": None, "progress_available": False}
        assert snapshot_files(store.folder) == before


@pytest.mark.parametrize("failure", ["snapshot", "console"])
def test_reporting_failure_does_not_retry_successful_work(store, monkeypatch, failure):
    import video_automation.progress as tracking
    store.progress_stream = io.StringIO()
    if failure == "console":
        store.progress_stream.close()
    else:
        def fail_save(*args):
            raise OSError("Progress disk fixture")
        monkeypatch.setattr(tracking, "write_json", fail_save)
    progress = WorkflowProgress(store, ["images"])
    store.progress = progress
    calls = []

    def generate():
        calls.append(1)
        return {"files": []}

    with progress.phase("images"):
        store.task("image.cue_0001", {}, generate)
        store.task("image.cue_0001", {}, generate)
    progress.finish("completed")
    assert calls == [1]
    assert store.manifest.stages["image.cue_0001"].status == "completed"
    assert store.manifest.stages["image.cue_0001"].attempts == 1


def test_cli_progress_uses_stderr_and_keeps_stdout_json(store, tmp_path, capsys):
    settings = tmp_path / "config.json"
    settings.write_text(store.config.model_dump_json(), encoding="utf-8")
    assert main(["--config", str(settings), "fixture", "--job-id", "progress_cli"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "completed"
    assert "[1/3 render] Started" in captured.err
    assert "[3/3 validate] Completed" in captured.err
    assert "3/3 clips" in captured.err
    snapshot = read_json(store.config.jobs_dir / "progress_cli/progress.json")
    assert snapshot["status"] == "completed"
    assert snapshot["completed_phases"] == snapshot["total_phases"] == 3
