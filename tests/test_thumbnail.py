import base64
import json

import httpx
import pytest
from PIL import Image

from test_pipeline import FakeProviders, image_bytes
from video_automation import media
from video_automation.cli import main
from video_automation.models import Config, Manifest
from video_automation.pipeline import Pipeline
from video_automation.progress import read_progress
from video_automation.providers import Providers, STYLE
from video_automation.store import PipelineError, Store, atomic_bytes, file_hash, read_json, write_json


TEXT = "You imagine a caregiver holding a child beside a fire."
TITLE = "Caregivers Beside the Fire"


@pytest.fixture
def pipeline(tmp_path):
    config = Config.load().model_copy(update={"jobs_dir": tmp_path / "jobs", "voice_id": "fixture-voice",
        "width": 640, "height": 360, "max_attempts": 2, "retry_backoff_seconds": 0, "image_interval_seconds": 0})
    source = tmp_path / "narration.txt"
    source.write_text(TEXT, encoding="utf-8")
    result = Pipeline.create(config, script_path=source)
    result.providers = FakeProviders(TEXT)
    return result


def snapshot(folder):
    return {str(path.relative_to(folder)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in folder.rglob("*") if path.is_file()}


def prepare(pipeline):
    return pipeline.script()


@pytest.mark.parametrize("mode", ["script", "production"])
def test_new_jobs_enable_thumbnails_and_existing_jobs_retain_selection(pipeline, tmp_path, mode):
    config = pipeline.config
    current = pipeline if mode == "script" else Pipeline.create(config, "Caregivers")
    store = current.store
    assert store.manifest.thumbnail_enabled
    old = read_json(store.path("manifest.json"))
    old.pop("thumbnail_enabled")
    write_json(store.path("manifest.json"), old)
    before = snapshot(store.folder)
    assert not Store(store.folder, config).manifest.thumbnail_enabled
    assert snapshot(store.folder) == before
    kwargs = {"script_path": tmp_path / "narration.txt"} if mode == "script" else {"topic": "Caregivers"}
    with pytest.raises(PipelineError, match="resume --thumbnail"):
        Pipeline.create(config, job_id=store.manifest.job_id, thumbnail_enabled=True, **kwargs)
    assert snapshot(store.folder) == before
    reused = Pipeline.create(config, job_id=store.manifest.job_id, **kwargs)
    assert reused.store.folder == store.folder and not reused.store.manifest.thumbnail_enabled


@pytest.mark.parametrize("provider,key,endpoint", [
    ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/responses"),
    ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1/responses"),
])
def test_thumbnail_provider_receives_full_context_and_saves_complete_prompt(pipeline, monkeypatch, provider, key, endpoint):
    store = pipeline.store
    store.config.text_provider = provider
    store.config.spending_ceiling_usd = 1
    store.config.request_cost_caps_usd = {"thumbnail": 0.1}
    monkeypatch.setenv(key, "fixture-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps({"headline": "Who cares?", "prompt": "An illustrative caregiver holding a child."})}]}]})

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    pipeline.providers = Providers(store)
    result = pipeline.thumbnail_plan(prepare(pipeline), TITLE)
    assert len(requests) == 1 and str(requests[0].url) == endpoint
    payload = json.loads(requests[0].content)
    context = payload["input"][1]["content"][0]["text"].split("\nContext: ", 1)[1]
    assert json.loads(context) == {"narration": TEXT, "title": TITLE}
    assert "tools" not in payload
    assert payload["text"]["format"]["strict"]
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False
    assert payload["text"]["format"]["schema"]["required"] == ["headline", "prompt"]
    assert result == read_json(store.path("thumbnail_prompt.json"))
    assert result["headline"] == "WHO CARES?"
    assert STYLE in result["prompt"] and '"WHO CARES?"' in result["prompt"]
    for required in ("BIG, BOLD, UPPERCASE", "TOP", "safe", "below", "No other text"):
        assert required in result["prompt"]
    assert store.manifest.usage[0]["operation"] == "thumbnail"
    assert store.manifest.usage[0]["reserved_upper_bound_usd"] == 0.1
    assert store.path(store.manifest.usage[0]["response_path"]).is_file()


@pytest.mark.parametrize("output,attempts", [
    ({"type": "output_text", "text": "not JSON"}, 2),
    ({"type": "output_text", "text": json.dumps({"headline": "Headline"})}, 2),
    ({"type": "output_text", "text": json.dumps({"headline": " ", "prompt": "Scene"})}, 2),
    ({"type": "output_text", "text": json.dumps({"headline": "Two\nlines", "prompt": "Scene"})}, 2),
    ({"type": "output_text", "text": json.dumps({"headline": "Headline", "prompt": " "})}, 2),
    ({"type": "output_text", "text": json.dumps({"headline": "Headline", "prompt": "Scene", "extra": True})}, 2),
    ({"type": "refusal", "refusal": "Fixture refusal"}, 1),
])
def test_bad_plans_and_refusals_are_reported_with_bounded_attempts(pipeline, monkeypatch, output, attempts):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    provider = Providers(pipeline.store)
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return {"status": "completed", "output": [{"type": "message", "content": [output]}]}

    monkeypatch.setattr(provider, "post", post)
    pipeline.providers = provider
    report = pipeline.thumbnail(prepare(pipeline), TITLE)
    assert report["status"] == "failed" and report["error"]
    assert len(calls) == attempts
    assert pipeline.store.manifest.stages["thumbnail.plan"].status == "failed"
    assert "thumbnail.image" not in pipeline.store.manifest.stages
    assert not pipeline.store.path("thumbnail_prompt.json").exists()


@pytest.mark.parametrize("caps,ceiling,error", [
    ({}, 1, "Missing conservative request_cost_caps_usd for thumbnail"),
    ({"thumbnail": 0.2}, 0.1, "Spending ceiling would be exceeded before thumbnail"),
])
def test_thumbnail_budget_failure_makes_no_network_request(pipeline, monkeypatch, caps, ceiling, error):
    pipeline.config.spending_ceiling_usd = ceiling
    pipeline.config.request_cost_caps_usd = caps
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("Network must not be reached"))
    pipeline.providers = Providers(pipeline.store)
    report = pipeline.thumbnail(prepare(pipeline), TITLE)
    assert report["status"] == "failed" and error in report["error"]
    assert pipeline.store.manifest.usage == []


@pytest.mark.parametrize("bad", ["missing", "corrupt", "small", "square"])
def test_external_thumbnail_handoff_never_consumes_image_attempts(pipeline, bad):
    store = pipeline.store
    if bad == "corrupt":
        atomic_bytes(store.path("thumbnail.png"), b"invalid")
    elif bad in {"small", "square"}:
        Image.new("RGB", (160, 90) if bad == "small" else (640, 640)).save(store.path("thumbnail.png"))
    for _ in range(2):
        report = pipeline.thumbnail(prepare(pipeline), TITLE)
        assert report["status"] == ("waiting_for_image" if bad == "missing" else "failed")
    assert pipeline.providers.calls == ["thumbnail"]
    assert "thumbnail.image" not in store.manifest.stages
    assert store.manifest.usage == []
    assert read_json(store.path("thumbnail_report.json")) == report


def test_external_adoption_and_replacement_are_independent_of_api_settings(pipeline):
    store = pipeline.store
    atomic_bytes(store.path("thumbnail.png"), image_bytes())
    before = store.path("thumbnail.png").read_bytes(), store.path("thumbnail.png").stat().st_mtime_ns
    assert pipeline.thumbnail(prepare(pipeline), TITLE)["status"] == "ready"
    assert before == (store.path("thumbnail.png").read_bytes(), store.path("thumbnail.png").stat().st_mtime_ns)
    pipeline.config.image_provider = "gemini"
    pipeline.config.gemini_model = "unused-model"
    assert pipeline.thumbnail(prepare(pipeline), TITLE)["status"] == "ready"
    assert store.manifest.stages["thumbnail.image"].attempts == 1
    Image.new("RGB", (640, 360), "blue").save(store.path("thumbnail.png"))
    replacement = file_hash(store.path("thumbnail.png"))
    pipeline.thumbnail(prepare(pipeline), TITLE)
    stage = store.manifest.stages["thumbnail.image"]
    assert stage.artifacts["thumbnail.png"] == replacement and stage.result["source"] == "external"
    assert pipeline.providers.calls == ["thumbnail"]


def test_api_uses_exact_saved_prompt_and_reuses_cached_thumbnail(pipeline):
    pipeline.store.manifest.image_mode = "api"
    prompts = []

    def image(prompt):
        assert prompt == read_json(pipeline.store.path("thumbnail_prompt.json"))["prompt"]
        prompts.append(prompt)
        return image_bytes()

    pipeline.providers.image = image
    for _ in range(2):
        assert pipeline.thumbnail(prepare(pipeline), TITLE)["status"] == "ready"
    assert len(prompts) == 1
    assert pipeline.store.manifest.stages["thumbnail.image"].attempts == 1
    assert read_json(pipeline.store.path("thumbnail_report.json"))["status"] == "ready"


def test_api_thumbnail_fallback_uses_saved_prompt_and_reserves_each_request(pipeline, monkeypatch):
    store = pipeline.store
    store.manifest.image_mode = "api"
    pipeline.config.spending_ceiling_usd = 1
    pipeline.config.request_cost_caps_usd = {"thumbnail": 0.1, "image": 0.2}
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter")
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini")
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps({"headline": "Who cares?", "prompt": "Illustrative caregiver and child."})}]}]})
        saved = read_json(store.path("thumbnail_prompt.json"))["prompt"]
        payload = json.loads(request.content)
        if request.url.path.endswith("/images"):
            assert payload["prompt"] == saved
            return httpx.Response(402, json={"error": {"message": "Insufficient credits"}})
        assert request.url.host == "generativelanguage.googleapis.com"
        assert payload["contents"][0]["parts"] == [{"text": saved}]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image_bytes()).decode()}}]}}]})

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    pipeline.providers = Providers(store)
    assert pipeline.thumbnail(prepare(pipeline), TITLE)["status"] == "ready"
    assert len(requests) == 3
    assert [entry["operation"] for entry in store.manifest.usage] == ["thumbnail", "image", "image"]
    assert [entry["reserved_upper_bound_usd"] for entry in store.manifest.usage] == [0.1, 0.2, 0.2]
    assert store.manifest.stages["thumbnail.image"].attempts == 1
    assert pipeline.thumbnail(prepare(pipeline), TITLE)["status"] == "ready"
    assert len(requests) == 3


def test_invalid_api_thumbnail_retries_and_image_model_only_invalidates_image(pipeline):
    pipeline.store.manifest.image_mode = "api"
    calls = []

    def image(prompt):
        calls.append(prompt)
        return b"invalid image"

    pipeline.providers.image = image
    assert pipeline.thumbnail(prepare(pipeline), TITLE)["status"] == "failed"
    assert len(calls) == 2
    pipeline.store.manifest.stages["thumbnail.image"].attempts = 0
    pipeline.providers.image = lambda prompt: calls.append(prompt) or image_bytes()
    assert pipeline.thumbnail(prepare(pipeline), TITLE)["status"] == "ready"
    pipeline.config.openrouter_image_model = "provider/new-image-model"
    assert pipeline.thumbnail(prepare(pipeline), TITLE)["status"] == "ready"
    assert len(calls) == 4 and pipeline.providers.calls == ["thumbnail"]


@pytest.mark.parametrize("change", ["narration", "title", "model", "prompt", "layout", "style", "cached_plan", "cached_scene", "artifact"])
def test_thumbnail_plan_cache_tracks_context_and_validates_cached_data(pipeline, monkeypatch, change):
    import video_automation.pipeline as orchestration
    script, title = prepare(pipeline), TITLE
    pipeline.thumbnail_plan(script, title)
    pipeline.thumbnail_plan(script, title)
    assert pipeline.providers.calls == ["thumbnail"]
    if change == "narration":
        script.narration += " An imagined camp."
    elif change == "title":
        title += " Again"
    elif change == "model":
        pipeline.config.openrouter_model = "provider/other-model"
    elif change in {"prompt", "layout", "style"}:
        name = {"prompt": "THUMBNAIL_PROMPT", "layout": "THUMBNAIL_LAYOUT", "style": "STYLE"}[change]
        monkeypatch.setattr(orchestration, name, getattr(orchestration, name) + "\nUpdated.")
    elif change == "cached_plan":
        pipeline.store.manifest.stages["thumbnail.plan"].result["data"]["headline"] = "lowercase"
    elif change == "cached_scene":
        data = pipeline.store.manifest.stages["thumbnail.plan"].result["data"]
        data["prompt"] = data["prompt"][data["prompt"].index("\n" + STYLE):]
    else:
        atomic_bytes(pipeline.store.path("thumbnail_prompt.json"), b"corrupt")
    result = pipeline.thumbnail_plan(script, title)
    assert pipeline.providers.calls == ["thumbnail", "thumbnail"]
    assert result["headline"] == result["headline"].upper()


@pytest.mark.parametrize("failure", ["plan", "image"])
def test_thumbnail_failure_does_not_block_real_video_and_retry_preserves_outputs(pipeline, monkeypatch, failure):
    from video_automation.store import TemporaryError
    store = pipeline.store
    store.manifest.image_mode = "api"
    original = pipeline.providers.thumbnail if failure == "plan" else pipeline.providers.image

    def fail(*args):
        raise TemporaryError("Thumbnail fixture failure")

    if failure == "plan":
        pipeline.providers.thumbnail = fail
    else:
        def image(prompt):
            if "video thumbnail" in prompt:
                return fail()
            return original(prompt)
        pipeline.providers.image = image
    assert pipeline.run()["passed"]
    assert store.manifest.status == "completed"
    assert read_progress(store.folder)["thumbnail"]["status"] == "failed"
    preserved = {str(path.relative_to(store.folder)): file_hash(path) for path in store.folder.rglob("*")
                 if path.is_file() and (path.suffix in {".mp4", ".wav", ".png", ".srt"} or path.name == "alignment.json")}
    stage = store.manifest.stages[f"thumbnail.{failure}"]
    assert stage.attempts == pipeline.config.max_attempts
    if failure == "plan":
        pipeline.providers.thumbnail = lambda *args: pytest.fail("Exhausted plan must not make another request")
    else:
        pipeline.providers.image = lambda *args: pytest.fail("Exhausted image must not make another request")
    assert pipeline.run()["passed"]
    if failure == "plan":
        pipeline.providers.thumbnail = original
    else:
        pipeline.providers.image = original
    monkeypatch.setattr("video_automation.pipeline.Providers", lambda store: pipeline.providers)
    settings = store.folder.parent / "config.json"
    settings.write_text(pipeline.config.model_dump_json(), encoding="utf-8")
    assert main(["--config", str(settings), "resume", store.manifest.job_id, "--retry-failed"]) == 0
    assert read_progress(store.folder)["thumbnail"]["status"] == "ready"
    assert preserved == {name: file_hash(store.path(name)) for name in preserved}


def test_missing_external_thumbnail_and_replacement_never_rebuild_video(pipeline, monkeypatch):
    store = pipeline.store
    assert pipeline.run()["status"] == "waiting_for_images"
    for cue in store.manifest.cues:
        atomic_bytes(store.path(f"images/{cue.id}.png"), image_bytes())
    assert pipeline.run()["passed"]
    assert read_progress(store.folder)["thumbnail"]["status"] == "waiting_for_image"
    preserved = {str(path.relative_to(store.folder)): (file_hash(path), path.stat().st_mtime_ns)
                 for path in store.folder.rglob("*") if path.is_file()
                 and (path.suffix in {".mp4", ".wav"} or path.parent.name == "images")}
    subtitles = file_hash(store.path("subtitles.en.srt"))
    calls = list(pipeline.providers.calls)
    for color in ("red", "blue"):
        Image.new("RGB", (640, 360), color).save(store.path("thumbnail.png"))
        assert pipeline.run()["passed"]
        assert read_progress(store.folder)["thumbnail"]["status"] == "ready"
        assert pipeline.providers.calls == calls
        assert preserved == {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns) for name in preserved}
        assert file_hash(store.path("subtitles.en.srt")) == subtitles
    import video_automation.pipeline as orchestration
    monkeypatch.setattr(orchestration, "THUMBNAIL_LAYOUT", orchestration.THUMBNAIL_LAYOUT + "\nKeep a simple background.")
    assert pipeline.run()["passed"]
    assert pipeline.providers.calls == calls + ["thumbnail"]
    assert preserved == {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns) for name in preserved}


@pytest.mark.parametrize("flag,enabled", [("--thumbnail", True), ("--no-thumbnail", False)])
def test_cli_new_run_thumbnail_flags(pipeline, tmp_path, monkeypatch, capsys, flag, enabled):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-text")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fixture-speech")
    monkeypatch.setattr("video_automation.pipeline.Providers", lambda store: pipeline.providers)
    settings = tmp_path / "config.json"
    settings.write_text(pipeline.config.model_dump_json(), encoding="utf-8")
    assert main(["--config", str(settings), "run", str(tmp_path / "narration.txt"), "--job-id", "new_cli", flag]) == 0
    result = json.loads(capsys.readouterr().out)
    assert ("thumbnail" in result) == enabled
    assert ("thumbnail" in pipeline.providers.calls) == enabled
    assert read_json(pipeline.config.jobs_dir / "new_cli" / "manifest.json")["thumbnail_enabled"] == enabled


@pytest.mark.parametrize("change", ["deleted_image", "corrupt_report"])
def test_status_handles_missing_thumbnail_and_corrupt_report_without_writes(pipeline, change):
    atomic_bytes(pipeline.store.path("thumbnail.png"), image_bytes())
    pipeline.thumbnail(prepare(pipeline), TITLE)
    if change == "deleted_image":
        pipeline.store.path("thumbnail.png").unlink()
    else:
        atomic_bytes(pipeline.store.path("thumbnail_report.json"), b"not JSON")
    before = snapshot(pipeline.store.folder)
    assert read_progress(pipeline.store.folder)["thumbnail"]["status"] == (
        "waiting_for_image" if change == "deleted_image" else "failed")
    assert snapshot(pipeline.store.folder) == before


def test_cli_enable_disable_and_status_are_persisted_and_read_only(pipeline, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("video_automation.pipeline.Providers", lambda store: pipeline.providers)
    settings = tmp_path / "config.json"
    settings.write_text(pipeline.config.model_dump_json(), encoding="utf-8")
    job = pipeline.store.manifest.job_id
    for flag, enabled in (("--no-thumbnail", False), ("--thumbnail", True)):
        assert main(["--config", str(settings), "resume", job, flag]) == 0
        report = json.loads(capsys.readouterr().out)
        assert ("thumbnail" in report) == enabled
        assert read_json(pipeline.store.path("manifest.json"))["thumbnail_enabled"] == enabled
        if enabled:
            assert report["thumbnail"]["prompt"] == str(pipeline.store.path("thumbnail_prompt.json"))
            assert report["thumbnail"]["image"] == str(pipeline.store.path("thumbnail.png"))
            before = snapshot(pipeline.store.folder)
            assert main(["--config", str(settings), "status", job]) == 0
            assert json.loads(capsys.readouterr().out)["thumbnail"]["status"] == "waiting_for_image"
            assert snapshot(pipeline.store.folder) == before


@pytest.mark.parametrize("mode", ["fixture", "imported"])
def test_fixture_and_imported_jobs_skip_thumbnail_even_if_enabled(pipeline, monkeypatch, mode):
    store = Store(pipeline.config.jobs_dir / mode, pipeline.config,
                  Manifest(job_id=mode, topic="Fixture", mode=mode, thumbnail_enabled=True))
    monkeypatch.setattr(media, "render", lambda store: None)
    monkeypatch.setattr(media, "finish", lambda store: None)
    monkeypatch.setattr(media, "validate", lambda store: {"passed": True})
    assert Pipeline(store, pipeline.providers).run()["passed"]
    assert pipeline.providers.calls == []
    assert not store.path("thumbnail_prompt.json").exists()
    assert "thumbnail" not in read_progress(store.folder)
    assert read_progress(store.folder)["total_phases"] == 3


def test_cli_fixture_cannot_enable_thumbnail_or_mutate_image_mode(pipeline, tmp_path, capsys):
    store = Store(pipeline.config.jobs_dir / "fixture", pipeline.config,
                  Manifest(job_id="fixture", topic="Fixture", mode="fixture"))
    settings = tmp_path / "config.json"
    settings.write_text(pipeline.config.model_dump_json(), encoding="utf-8")
    before = snapshot(store.folder)
    assert main(["--config", str(settings), "resume", "fixture", "--thumbnail", "--image-mode", "api"]) == 1
    assert "only for script and topic jobs" in capsys.readouterr().err
    assert snapshot(store.folder) == before


def test_thumbnail_keyboard_interrupt_propagates_and_preserves_interrupted_status(pipeline):
    def interrupt(*args):
        raise KeyboardInterrupt()
    pipeline.providers.thumbnail = interrupt
    with pytest.raises(KeyboardInterrupt):
        pipeline.run()
    assert pipeline.store.manifest.status == "interrupted"
    assert read_progress(pipeline.store.folder)["phase"] == "thumbnail"
