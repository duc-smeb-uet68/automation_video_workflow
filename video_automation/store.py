from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .models import Manifest, Stage


class PipelineError(RuntimeError):
    pass


class TemporaryError(PipelineError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, content):
    atomic_bytes(path, (json.dumps(content, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@contextmanager
def worker_lock(root: Path):
    """An OS-released lock survives crashes without leaving a stale lock state."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise PipelineError("Another worker already owns this jobs directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Store:
    def __init__(self, folder: Path, config, manifest: Manifest | None = None):
        self.folder = folder.resolve()
        self.config = config
        self.manifest = manifest or Manifest.model_validate(read_json(self.folder / "manifest.json"))
        if self.manifest.mode in {"production", "script"} and self.manifest.image_strategy != "text_only":
            raise PipelineError("Legacy reference-based job is preserved; create a new text-only job on this branch "
                                "or use main to continue the existing job")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.save()

    def path(self, relative: str) -> Path:
        path = (self.folder / relative).resolve()
        if not path.is_relative_to(self.folder):
            raise PipelineError(f"Artifact escapes job directory: {relative}")
        return path

    def save(self):
        write_json(self.folder / "manifest.json", self.manifest.model_dump(mode="json"))

    def log(self, event: str, **details):
        with (self.folder / "execution.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), "event": event, **details}) + "\n")

    def task(self, name: str, inputs, action):
        key = digest(inputs)
        previous = self.manifest.stages.get(name)
        intact = previous and all(self.path(p).is_file() and file_hash(self.path(p)) == h
                                  for p, h in previous.artifacts.items())
        if previous and previous.input_hash == key and previous.status == "completed" and intact:
            self.log("cache_hit", stage=name)
            return previous.result
        if previous and previous.input_hash == key and previous.status != "completed":
            stage = previous
        else:
            stage = Stage(input_hash=key)
            self.manifest.stages[name] = stage
        while stage.attempts < self.config.max_attempts:
            stage.status = "running"
            stage.attempts += 1
            self.save()
            self.log("stage_start", stage=name, attempt=stage.attempts)
            try:
                result = action()
                stage.artifacts = {p: file_hash(self.path(p)) for p in result.get("files", [])}
                stage.result = result
                stage.status = "completed"
                self.save()
                self.log("stage_complete", stage=name)
                return result
            except Exception as exc:
                stage.status = "failed"
                stage.errors.append(str(exc))
                self.save()
                self.log("stage_failed", stage=name, error=str(exc))
                if not isinstance(exc, TemporaryError) or stage.attempts >= self.config.max_attempts:
                    raise
                time.sleep(min(30, self.config.retry_backoff_seconds * 2 ** (stage.attempts - 1)))
        raise PipelineError(f"{name}: retry limit reached; use resume --retry-failed for an explicit new retry batch")

    def reserve_request(self, operation: str):
        cap = self.config.request_cost_caps_usd.get(operation)
        ceiling = self.config.spending_ceiling_usd
        if ceiling is not None:
            if cap is None:
                raise PipelineError(f"Missing conservative request_cost_caps_usd for {operation}")
            spent = sum(item.get("reserved_upper_bound_usd") or 0 for item in self.manifest.usage)
            if spent + cap > ceiling:
                raise PipelineError(f"Spending ceiling would be exceeded before {operation}")
        entry = {"operation": operation, "reserved_upper_bound_usd": cap, "status": "pending"}
        self.manifest.usage.append(entry)
        self.save()
        return entry
