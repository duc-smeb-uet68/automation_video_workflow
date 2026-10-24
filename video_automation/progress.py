from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone

from .store import read_json, write_json


class WorkflowProgress:
    """Report phase and checkpoint progress without changing task execution."""

    def __init__(self, store, phases):
        self.store = store
        self.stream = store.progress_stream
        self.started = time.monotonic()
        self.seen_tasks = set()
        self.warned = False
        self.state = {"job_id": store.manifest.job_id, "mode": store.manifest.mode,
                      "status": "running", "completed_phases": 0, "total_phases": len(phases),
                      "phase": None, "phase_number": 0, "phase_status": "pending",
                      "completed_units": 0, "cached_units": 0, "total_units": None, "unit": None,
                      "task": None, "task_status": None, "attempt": None,
                      "max_attempts": store.config.max_attempts, "error": None}

    def console(self, message):
        if self.stream is not None:
            try:
                print(message, file=self.stream, flush=True)
            except (OSError, ValueError):
                # A disconnected terminal must not cause another paid request.
                self.stream = None

    def publish(self, message):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        try:
            write_json(self.store.path("progress.json"), self.state)
        except OSError:
            if not self.warned:
                self.console("Warning: could not save progress.json; workflow checkpoints remain authoritative.")
                self.warned = True
        prefix = f"[{self.state['job_id']}] [{self.state['phase_number']}/{self.state['total_phases']}"
        if self.state["phase"]:
            prefix += f" {self.state['phase']}"
        units = ""
        if self.state["total_units"] is not None:
            units = f" | {self.state['completed_units']}/{self.state['total_units']} {self.state['unit']}"
        self.console(f"{prefix}] {message}{units}")

    @contextmanager
    def phase(self, name):
        self.seen_tasks.clear()
        self.state.update(phase=name, phase_number=self.state["completed_phases"] + 1,
                          phase_status="running", completed_units=0, cached_units=0,
                          total_units=None, unit=None, task=None, task_status=None, attempt=None)
        if name in {"images", "render"}:
            self.state.update(total_units=len(self.store.manifest.cues), unit="images" if name == "images" else "clips")
        self.publish("Started")
        yield
        self.state["completed_phases"] += 1
        self.state.update(phase_status="completed", task=None, task_status=None, attempt=None)
        self.publish("Completed")

    def set_total(self, total, unit):
        self.state.update(total_units=total, unit=unit)
        self.publish("Work planned")

    def on_event(self, event, details):
        statuses = {"stage_start": "running", "stage_complete": "completed",
                    "cache_hit": "cached", "stage_failed": "failed", "stage_exhausted": "failed"}
        if event not in statuses:
            return
        task = details["stage"]
        stage = self.store.manifest.stages.get(task)
        self.state.update(task=task, task_status=statuses[event],
                          attempt=details.get("attempt", stage.attempts if stage else None))
        phase = self.state["phase"]
        counted = ((phase == "visuals" and task.startswith("visuals."))
                   or (phase == "images" and task.startswith("image.") and task != "image.prompts")
                   or (phase == "render" and task.startswith("render.")))
        if counted and event in {"stage_complete", "cache_hit"} and task not in self.seen_tasks:
            self.seen_tasks.add(task)
            self.state["completed_units"] += 1
            if event == "cache_hit":
                self.state["cached_units"] += 1
        attempt = self.state["attempt"]
        suffix = f" (attempt {attempt}/{self.state['max_attempts']})" if event in {"stage_start", "stage_failed", "stage_exhausted"} else ""
        self.publish(f"{task}: {statuses[event]}{suffix}")

    def finish(self, status, error=None):
        self.state.update(status=status, error=error)
        if status != "completed":
            self.state["phase_status"] = status
        self.publish(status.capitalize())


def read_thumbnail(folder, manifest):
    """Expose thumbnail readiness independently of video status, without writing files."""
    if not manifest.get("thumbnail_enabled", False) or manifest.get("mode") not in {"production", "script"}:
        return None
    report = {"status": "waiting_for_image", "error": None}
    path = folder / "thumbnail_report.json"
    if path.is_file():
        try:
            saved = read_json(path)
            if not isinstance(saved, dict) or saved.get("status") not in {"waiting_for_image", "ready", "failed"}:
                raise ValueError("Invalid thumbnail report")
            report.update(status=saved["status"], error=saved.get("error"))
            if report["status"] == "ready" and not (folder / "thumbnail.png").is_file():
                report.update(status="waiting_for_image", error="Thumbnail file is missing; resume to validate readiness.")
        except (OSError, ValueError):
            report.update(status="failed", error="Thumbnail report is unreadable; resume to refresh it.")
    return {**report, "prompt": str(folder / "thumbnail_prompt.json"), "image": str(folder / "thumbnail.png"),
            "report": str(path)}


def read_progress(folder):
    """Read a job snapshot without taking the worker lock or writing any files."""
    manifest = read_json(folder / "manifest.json")
    path = folder / "progress.json"
    progress = read_json(path) if path.is_file() else {}
    result = {**progress, "job_id": manifest["job_id"], "mode": manifest["mode"],
            "image_mode": manifest.get("image_mode", "api"),
            "status": manifest["status"], "error": manifest.get("error"), "progress_available": bool(progress)}
    thumbnail = read_thumbnail(folder, manifest)
    if thumbnail is not None:
        result["thumbnail"] = thumbnail
    return result
