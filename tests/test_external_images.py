import json

import pytest
from PIL import Image

from test_pipeline import FakeProviders, image_bytes, narration
from video_automation import media
from video_automation.cli import doctor, main
from video_automation.models import Config, Cue, Manifest
from video_automation.pipeline import ImagesPending, Pipeline
from video_automation.progress import read_progress
from video_automation.providers import Providers
from video_automation.store import PipelineError, Store, atomic_bytes, file_hash, read_json, write_json


@pytest.fixture
def config(tmp_path):
    return Config.load().model_copy(update={"jobs_dir": tmp_path / "jobs", "width": 640, "height": 360,
        "voice_id": "fixture-voice", "max_attempts": 2, "retry_backoff_seconds": 0, "image_interval_seconds": 0})


def create_pipeline(config, tmp_path, mode="script"):
    if mode == "script":
        source = tmp_path / "narration.txt"
        source.write_text(narration(), encoding="utf-8")
        pipeline = Pipeline.create(config, script_path=source)
    else:
        pipeline = Pipeline.create(config, "External images fixture")
    pipeline.providers = FakeProviders()
    return pipeline


def test_external_mode_is_default_and_api_mode_requires_selected_image_key():
    config = Config(text_provider="openai", image_provider="gemini")
    assert config.image_mode == "external"
    assert config.generation_keys == ("OPENAI_API_KEY", "ELEVENLABS_API_KEY")
    config.image_mode = "api"
    assert config.generation_keys == ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "GEMINI_API_KEY")


def test_doctor_external_mode_ignores_image_credentials(config, monkeypatch, capsys):
    config.text_provider = "openai"
    config.image_provider = "gemini"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "invalid unused key array [")
    monkeypatch.setenv("OPENAI_API_KEY", "text-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "speech-key")
    monkeypatch.setattr(media, "run", lambda command: "version")
    assert doctor(config) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["image_mode"] == "external"
    assert report["image_provider"] is None and report["image_fallback"] is None
    assert "OPENROUTER_API_KEY" not in report["checks"] and "GEMINI_API_KEY" not in report["checks"]
    assert "text-key" not in json.dumps(report)


@pytest.mark.parametrize("mode", ["script", "topic"])
def test_preparation_waits_with_all_handoff_files_and_no_image_attempts(config, tmp_path, mode):
    pipeline = create_pipeline(config, tmp_path, mode)
    provider = pipeline.providers
    result = pipeline.run()
    store = pipeline.store
    assert result["status"] == store.manifest.status == "waiting_for_images"
    assert store.manifest.image_mode == "external" and store.manifest.error is None
    assert store.manifest.total_frames > 0
    for path in ("script.txt", "script.json", "audio/narration.wav", "alignment.json", "transcript.json",
                 "image_prompts.json", "subtitles.en.srt", "external_images.json"):
        assert store.path(path).is_file(), path
    assert store.path("images").is_dir()
    assert not list(store.path("images").iterdir())
    assert not store.path("rough.mp4").exists() and not store.path("final.mp4").exists()
    assert not store.path("failure_report.json").exists()
    assert "image" not in provider.calls
    assert not any(name.startswith("image.cue_") for name in store.manifest.stages)
    assert len(result["missing_images"]) == len(store.manifest.cues)
    progress = read_progress(store.folder)
    assert progress["status"] == progress["phase_status"] == "waiting_for_images"
    assert progress["phase"] == "images" and progress["completed_phases"] == 3
    before = list(provider.calls)
    resumed = Pipeline(Store(store.folder, config), provider)
    assert resumed.run()["status"] == "waiting_for_images"
    assert provider.calls == before
    assert all(stage.attempts == 1 for stage in resumed.store.manifest.stages.values())


def external_store(config):
    cues = [Cue(id=f"cue_{index:04}", text="A scene.", start=index-1, end=index, word_start=index-1,
                word_end=index, scene_description=f"Scene {index}.") for index in (1, 2)]
    return Store(config.jobs_dir / "files", config, Manifest(job_id="files", topic="External images",
                 image_mode="external", cues=cues))


@pytest.mark.parametrize("bad_image", ["corrupt", "too_small", "wrong_shape"])
def test_partial_invalid_images_wait_without_retries_and_preserve_valid_file(config, bad_image):
    store = external_store(config)
    first, second = store.path("images/cue_0001.png"), store.path("images/cue_0002.png")
    atomic_bytes(first, image_bytes())
    if bad_image == "corrupt":
        atomic_bytes(second, b"not an image")
    else:
        Image.new("RGB", (160, 90) if bad_image == "too_small" else (640, 640)).save(second)
    before = first.read_bytes(), first.stat().st_mtime_ns, second.read_bytes(), second.stat().st_mtime_ns
    pipeline = Pipeline(store, FakeProviders())
    for _ in range(3):
        with pytest.raises(ImagesPending) as pending:
            pipeline.images()
        assert pending.value.report["ready_images"] == 1
        assert pending.value.report["invalid_images"][0]["path"] == "images/cue_0002.png"
    assert before == (first.read_bytes(), first.stat().st_mtime_ns, second.read_bytes(), second.stat().st_mtime_ns)
    assert store.manifest.stages["image.cue_0001"].attempts == 1
    assert "image.cue_0002" not in store.manifest.stages
    assert pipeline.providers.calls == [] and store.manifest.usage == []
    atomic_bytes(second, image_bytes())
    pipeline.images()
    assert read_json(store.path("external_images.json"))["status"] == "ready"
    assert all(cue.image_path for cue in store.manifest.cues)


def test_external_replacements_are_adopted_and_api_settings_do_not_invalidate_them(config):
    store = external_store(config)
    for cue in store.manifest.cues:
        atomic_bytes(store.path(f"images/{cue.id}.png"), image_bytes())
    provider = FakeProviders()
    Pipeline(store, provider).images()
    original = file_hash(store.path("images/cue_0001.png"))
    Image.new("RGB", (640, 360), "blue").save(store.path("images/cue_0001.png"))
    config.openrouter_image_model = "unused-model"
    config.image_provider = "gemini"
    resumed = Store(store.folder, config)
    Pipeline(resumed, provider).images()
    replacement = file_hash(store.path("images/cue_0001.png"))
    assert replacement != original
    assert resumed.manifest.stages["image.cue_0001"].artifacts["images/cue_0001.png"] == replacement
    assert resumed.manifest.stages["image.cue_0001"].result["source"] == "external"
    assert resumed.manifest.stages["image.cue_0002"].attempts == 1
    assert provider.calls == []


def test_provider_boundary_disables_image_requests_for_external_jobs(config, monkeypatch):
    client = Providers(external_store(config))
    monkeypatch.setattr(client, "post", lambda *args: pytest.fail("External jobs must never request image APIs"))
    with pytest.raises(PipelineError, match="Image API calls are disabled"):
        client.image("A scene.")


def test_existing_api_jobs_keep_mode_and_external_job_identity_is_distinct(config, tmp_path):
    config.image_mode = "api"
    old = create_pipeline(config, tmp_path)
    saved = read_json(old.store.path("manifest.json"))
    saved.pop("image_mode")
    write_json(old.store.path("manifest.json"), saved)
    before = old.store.path("manifest.json").read_bytes()
    config.image_mode = "external"
    external = create_pipeline(config, tmp_path)
    assert external.store.folder != old.store.folder
    assert old.store.path("manifest.json").read_bytes() == before
    assert Store(old.store.folder, config).manifest.image_mode == "api"
    assert old.store.path("manifest.json").read_bytes() == before


def test_explicit_job_id_mode_collision_does_not_rewrite_old_job(config, tmp_path):
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    config.image_mode = "api"
    pipeline = Pipeline.create(config, script_path=source, job_id="existing")
    saved = read_json(pipeline.store.path("manifest.json"))
    saved.pop("image_mode")
    write_json(pipeline.store.path("manifest.json"), saved)
    before = pipeline.store.path("manifest.json").read_bytes(), pipeline.store.path("manifest.json").stat().st_mtime_ns
    config.image_mode = "external"
    with pytest.raises(PipelineError, match="resume --image-mode"):
        Pipeline.create(config, script_path=source, job_id="existing")
    assert before == (pipeline.store.path("manifest.json").read_bytes(), pipeline.store.path("manifest.json").stat().st_mtime_ns)


@pytest.mark.parametrize("mode", ["external", "api"])
def test_cli_run_mode_override_wins_over_config_and_persists(config, tmp_path, monkeypatch, capsys, mode):
    config.image_mode = "api" if mode == "external" else "external"
    for name in ("OPENROUTER_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.setenv(name, "fixture-key")
    settings = tmp_path / "config.json"
    settings.write_text(config.model_dump_json(), encoding="utf-8")
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    monkeypatch.setattr(Pipeline, "run", lambda pipeline: None)
    assert main(["--config", str(settings), "run", str(source), "--image-mode", mode]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["image_mode"] == mode
    assert read_json(config.jobs_dir / output["job_id"] / "manifest.json")["image_mode"] == mode


def test_cli_waits_without_image_key_then_resumes_supplied_images(config, tmp_path, monkeypatch, capsys):
    config.text_provider = "openai"
    config.image_provider = "gemini"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-text")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fixture-speech")
    settings = tmp_path / "config.json"
    settings.write_text(config.model_dump_json(), encoding="utf-8")
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    provider = FakeProviders()
    monkeypatch.setattr("video_automation.pipeline.Providers", lambda store: provider)
    assert main(["--config", str(settings), "run", str(source), "--job-id", "cli_external"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "waiting_for_images" and "video" not in output
    store = Store(config.jobs_dir / output["job_id"], config)
    for cue in store.manifest.cues:
        atomic_bytes(store.path(f"images/{cue.id}.png"), image_bytes())
    before_calls = list(provider.calls)
    monkeypatch.setattr(media, "render", lambda store: None)
    monkeypatch.setattr(media, "finish", lambda store: None)
    monkeypatch.setattr(media, "validate", lambda store: {"passed": True})
    assert main(["--config", str(settings), "resume", output["job_id"]]) == 0
    completed = json.loads(capsys.readouterr().out)
    assert completed["status"] == "completed" and completed["image_mode"] == "external"
    assert provider.calls == before_calls


@pytest.mark.parametrize("mode", ["external", "api"])
def test_cli_mode_override_is_persisted(config, tmp_path, monkeypatch, capsys, mode):
    settings = tmp_path / "config.json"
    settings.write_text(config.model_dump_json(), encoding="utf-8")
    pipeline = create_pipeline(config, tmp_path)
    monkeypatch.setattr(Pipeline, "run", lambda pipeline: None)
    assert main(["--config", str(settings), "resume", pipeline.store.manifest.job_id, "--image-mode", mode]) == 0
    assert read_json(pipeline.store.path("manifest.json"))["image_mode"] == mode


def test_external_images_render_real_video_and_replacements_refresh_clips(config, tmp_path):
    pipeline = create_pipeline(config, tmp_path)
    assert pipeline.run()["status"] == "waiting_for_images"
    provider, store = pipeline.providers, pipeline.store
    preserved = {path: file_hash(store.path(path)) for path in ("audio/narration.wav", "alignment.json", "image_prompts.json")}
    for cue in store.manifest.cues:
        atomic_bytes(store.path(f"images/{cue.id}.png"), image_bytes())
    before = list(provider.calls)
    resumed = Pipeline(Store(store.folder, config), provider)
    assert resumed.run()["passed"]
    assert provider.calls == before and "image" not in provider.calls
    assert preserved == {path: file_hash(store.path(path)) for path in preserved}
    video = file_hash(store.path("final.mp4"))
    assert resumed.run()["passed"]
    assert file_hash(store.path("final.mp4")) == video
    first = resumed.store.manifest.cues[0]
    old_clip = file_hash(store.path(f"render/clips/{first.id}.mp4"))
    Image.new("RGB", (640, 360), "blue").save(store.path(first.image_path))
    assert resumed.run()["passed"]
    assert file_hash(store.path(f"render/clips/{first.id}.mp4")) != old_clip
    assert file_hash(store.path("final.mp4")) != video
    assert provider.calls == before
