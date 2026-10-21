from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import assets, media
from .models import Config
from .pipeline import Pipeline
from .store import PipelineError, Store, worker_lock, write_json


def load_credentials(path):
    """Read only provider keys from an optional local .env; never print values."""
    if not path.is_file():
        return
    allowed = {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "GEMINI_API_KEY"}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip() in allowed:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value and not os.environ.get(key.strip()):
                os.environ[key.strip()] = value


def job_folder(config, value):
    path = Path(value)
    if path.is_file() and path.name == "manifest.json":
        return path.parent.resolve()
    if path.is_dir():
        return path.resolve()
    result = (config.jobs_dir / value).resolve()
    if not result.is_relative_to(config.jobs_dir.resolve()):
        raise PipelineError("Job name escapes jobs directory")
    return result


def doctor(config):
    checks = {}
    for name in ("ffmpeg", "ffprobe"):
        try:
            media.run([getattr(config, name), "-version"])
            checks[name] = True
        except (OSError, PipelineError):
            checks[name] = False
    for name in config.generation_keys:
        checks[name] = bool(os.environ.get(name))
    checks["voice_id"] = bool(config.voice_id)
    checks["music_file"] = not config.music_path or config.music_path.is_file()
    print(json.dumps({"ready_for_live_generation": all(checks.values()),
                      "text_provider": config.text_provider, "text_model": config.text_model,
                      "image_provider": config.image_provider, "image_model": config.image_model, "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resumable educational video production")
    parser.add_argument("--config", type=Path, help="JSON setup file; defaults to config.local.json when present")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check local tools and credential presence without revealing secrets")
    setup = commands.add_parser("init", help="Write setup configuration without overwriting an existing file")
    setup.add_argument("--output", type=Path, default=Path("config.local.json"))
    run = commands.add_parser("run", help="Produce a video from a narration file (default) or an explicit topic")
    run.add_argument("input", nargs="?", help="UTF-8 narration file, or topic text with --mode topic")
    run.add_argument("--mode", choices=("script", "topic"), help="Input mode; defaults to input_mode in configuration (script)")
    run.add_argument("--script", type=Path, help="UTF-8 narration file; selects script mode without a positional input")
    run.add_argument("--job-id")
    resume = commands.add_parser("resume", help="Resume an existing job with successful checkpoints preserved")
    resume.add_argument("job")
    resume.add_argument("--retry-failed", action="store_true", help="Explicitly grant failed stages one new bounded retry batch")
    validation = commands.add_parser("validate", help="Validate final video and write quality_report.json")
    validation.add_argument("job")
    test = commands.add_parser("fixture", help="Render a portable synthetic smoke test without provider calls")
    test.add_argument("--job-id", default="renderer_fixture")
    importer = commands.add_parser("import-assets", help="Import legacy image manifest, narration and SRT, then render")
    importer.add_argument("--manifest", type=Path, required=True)
    importer.add_argument("--audio", type=Path, required=True)
    importer.add_argument("--subtitles", type=Path, required=True)
    importer.add_argument("--job-id", required=True)
    importer.add_argument("--limit-cues", type=int, help="Short renderer fixture only; skips production length acceptance")
    editor = commands.add_parser("export-kdenlive", help="Export an optional editable project using the existing workspace template")
    editor.add_argument("job")
    editor.add_argument("--template", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config_path = args.config or (Path("config.local.json") if Path("config.local.json").is_file() else None)
        config = Config.load(config_path)
        load_credentials((config_path.resolve().parent if config_path else Path.cwd()) / ".env")
        if args.command == "init":
            if args.output.exists():
                raise PipelineError(f"Configuration already exists: {args.output}")
            data = config.model_dump(mode="json", exclude={"character_reference"})
            data["jobs_dir"] = "jobs"
            write_json(args.output, data)
            print(str(args.output.resolve()))
            return 0
        if args.command == "doctor":
            return doctor(config)
        with worker_lock(config.jobs_dir):
            if args.command == "run":
                mode = args.mode or config.input_mode
                if args.script is not None:
                    if args.input is not None or args.mode == "topic":
                        raise PipelineError("Use --script alone, without a positional input or --mode topic")
                    mode = "script"
                if args.input is None and args.script is None:
                    raise PipelineError("run requires a narration file; use --script PATH or --mode topic TOPIC")
                if mode == "script":
                    pipeline = Pipeline.create(config, job_id=args.job_id, script_path=args.script or Path(args.input))
                else:
                    pipeline = Pipeline.create(config, args.input, args.job_id)
                if not pipeline.store.manifest.stages:
                    missing = [name for name in config.generation_keys if not os.environ.get(name)]
                    if not config.voice_id:
                        missing.append("voice_id in configuration")
                    if missing:
                        error = "Live generation requires setup: " + ", ".join(missing) + ". Run doctor after configuring locally."
                        pipeline.store.manifest.status, pipeline.store.manifest.error = "failed", error
                        pipeline.store.save()
                        write_json(pipeline.store.path("failure_report.json"), {"error": error, "stage_status": {}})
                        raise PipelineError(error)
            elif args.command == "fixture":
                pipeline = assets.fixture(config, args.job_id)
            elif args.command == "import-assets":
                pipeline = assets.import_assets(config, args.manifest.resolve(), args.audio.resolve(), args.subtitles.resolve(),
                                                args.job_id, args.limit_cues)
            else:
                store = Store(job_folder(config, args.job), config)
                if args.command == "validate":
                    try:
                        report = media.validate(store)
                    except PipelineError as exc:
                        store.manifest.status, store.manifest.error = "failed", str(exc)
                        store.save()
                        raise
                    print(json.dumps({"passed": report["passed"], "report": str(store.path("quality_report.json"))}, indent=2))
                    return 0
                if args.command == "export-kdenlive":
                    from .kdenlive import export
                    print(str(export(store, args.template)))
                    return 0
                if args.retry_failed:
                    for stage in store.manifest.stages.values():
                        if stage.status in {"failed", "running"}:
                            stage.attempts = 0
                    store.log("explicit_retry_batch")
                    store.save()
                pipeline = Pipeline(store)
            pipeline.run()
            print(json.dumps({"status": pipeline.store.manifest.status, "mode": pipeline.store.manifest.mode,
                              "video": str(pipeline.store.path("final.mp4")), "subtitles": str(pipeline.store.path("subtitles.en.srt")),
                              "report": str(pipeline.store.path("quality_report.json"))}, indent=2))
            return 0
    except KeyboardInterrupt:
        print("Interrupted; resume the job to continue from completed checkpoints.", file=sys.stderr)
        return 130
    except (PipelineError, OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
