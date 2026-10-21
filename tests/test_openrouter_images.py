import base64
import io
import json

import httpx
import pytest
from PIL import Image

from video_automation import media
from video_automation.cli import doctor, main
from video_automation.models import Config, Cue, Manifest
from video_automation.pipeline import Pipeline
from video_automation.providers import Providers, STYLE
from video_automation.store import PipelineError, Store, TemporaryError, atomic_bytes, file_hash, read_json


def png(color="#fffaf0"):
    buffer = io.BytesIO()
    Image.new("RGB", (640, 360), color).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    config = Config.load().model_copy(update={"jobs_dir": tmp_path / "jobs", "voice_id": "fixture-voice",
        "max_attempts": 2, "retry_backoff_seconds": 0, "image_interval_seconds": 0})
    return Store(config.jobs_dir / "images", config, Manifest(job_id="images", topic="Caregivers"))


def mock_http(monkeypatch, handler):
    client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))


def image_response(content):
    return {"data": [{"b64_json": base64.b64encode(content).decode("ascii"), "media_type": "image/png"}],
            "usage": {"cost": 0.012, "total_tokens": 1290}}


def test_requested_image_default():
    config = Config()
    assert config.image_provider == "openrouter"
    assert config.image_model == "google/gemini-3.1-flash-lite-image"
    assert config.image_key_env == "OPENROUTER_API_KEY"
    assert config.generation_keys == ("OPENROUTER_API_KEY", "ELEVENLABS_API_KEY")


@pytest.mark.parametrize("text,image,keys", [
    ("openrouter", "openrouter", ("OPENROUTER_API_KEY", "ELEVENLABS_API_KEY")),
    ("openrouter", "gemini", ("OPENROUTER_API_KEY", "ELEVENLABS_API_KEY", "GEMINI_API_KEY")),
    ("openai", "openrouter", ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "OPENROUTER_API_KEY")),
    ("openai", "gemini", ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "GEMINI_API_KEY")),
])
def test_generation_keys_follow_both_providers(text, image, keys):
    assert Config(text_provider=text, image_provider=image).generation_keys == keys


@pytest.mark.parametrize("reference_count", [0, 2])
def test_image_http_contract_references_and_usage(store, monkeypatch, reference_count):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    references = []
    for index, color in enumerate(["red", "blue"][:reference_count]):
        path = store.path(f"references/{index}.png")
        atomic_bytes(path, png(color))
        references.append(path)
    content, captured = png(), []
    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=image_response(content))
    mock_http(monkeypatch, handler)
    assert Providers(store).image("A camp scene.", references) == content
    request = captured[0]
    payload = json.loads(request.content)
    assert request.method == "POST" and str(request.url) == "https://openrouter.ai/api/v1/images"
    assert request.headers["Authorization"] == "Bearer fixture-openrouter-key"
    assert payload["model"] == "google/gemini-3.1-flash-lite-image"
    assert payload["prompt"] == "A camp scene.\n" + STYLE
    assert payload["n"] == 1 and payload["resolution"] == "1K" and payload["aspect_ratio"] == "16:9"
    if references:
        assert [base64.b64decode(r["image_url"]["url"].split(",", 1)[1]) for r in payload["input_references"]] == [p.read_bytes() for p in references]
        assert all(r["type"] == "image_url" for r in payload["input_references"])
    else:
        assert "input_references" not in payload
    usage = store.manifest.usage[0]
    assert usage["provider"] == "openrouter.ai" and usage["model"] == payload["model"]
    assert usage["provider_usage"]["cost"] == 0.012
    assert "fixture-openrouter-key" not in store.path("manifest.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("data", [
    {}, {"data": []}, {"data": [None]}, {"data": [{}]},
    {"data": [{"b64_json": "invalid-base64"}]}, {"data": [{"b64_json": None}]},
])
def test_missing_or_malformed_image_is_retryable(store, monkeypatch, data):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    monkeypatch.setattr(client, "post", lambda *args: data)
    with pytest.raises(TemporaryError):
        client.image("A scene", [])


def test_vector_image_is_rejected(store, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    monkeypatch.setattr(client, "post", lambda *args: {"data": [{"media_type": "image/svg+xml", "b64_json": "PHN2Zy8+"}]})
    with pytest.raises(PipelineError, match="unsupported image format"):
        client.image("A scene", [])


def test_gemini_direct_route_remains_available(store, monkeypatch):
    store.config.image_provider = "gemini"
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    content, captured = png(), []
    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"inlineData": {
            "mimeType": "image/png", "data": base64.b64encode(content).decode()}}]}}]})
    mock_http(monkeypatch, handler)
    assert Providers(store).image("A scene", []) == content
    request = captured[0]
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-image:generateContent"
    assert request.headers["x-goog-api-key"] == "fixture-gemini-key"


def test_doctor_and_cli_accept_openrouter_images_without_gemini_key(store, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fixture-key")
    monkeypatch.setattr(media, "run", lambda command: "version")
    assert doctor(store.config) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["image_provider"] == "openrouter"
    assert report["image_model"] == "google/gemini-3.1-flash-lite-image"
    assert "GEMINI_API_KEY" not in report["checks"]
    settings = tmp_path / "config.json"
    settings.write_text(store.config.model_dump_json(), encoding="utf-8")
    calls = []
    monkeypatch.setattr(Pipeline, "run", lambda pipeline: calls.append(1))
    assert main(["--config", str(settings), "run", "--mode", "topic", "Image preflight", "--job-id", "preflight"]) == 0
    assert calls == [1]


def test_failed_openrouter_image_resumes_without_replacing_successful_work(store, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    store.manifest.cues = [Cue(id=f"cue_{index:04}", text="A scene.", start=index-1, end=index,
        word_start=(index-1)*2, word_end=index*2, scene_group="camp", scene_description=f"Scene {index}.") for index in range(1, 4)]
    atomic_bytes(store.path("audio/narration.wav"), b"existing voiceover")
    voice_hash = file_hash(store.path("audio/narration.wav"))
    requests, failures_enabled = [], [True]
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if failures_enabled[0] and "Scene 2." in payload["prompt"]:
            return httpx.Response(429, json={"error": {"message": "fixture rate limit"}})
        return httpx.Response(200, json=image_response(png()))
    mock_http(monkeypatch, handler)
    with pytest.raises(TemporaryError, match="HTTP 429"):
        Pipeline(store).images()
    first_hash = file_hash(store.path("images/cue_0001.png"))
    assert len(requests) == 4  # Reference, first cue, then two bounded failed attempts.
    resumed = Store(store.folder, store.config)
    with pytest.raises(PipelineError, match="retry limit reached"):
        Pipeline(resumed).images()
    assert len(requests) == 4
    failures_enabled[0] = False
    resumed.manifest.stages["image.cue_0002"].attempts = 0
    resumed.save()
    Pipeline(resumed).images()
    assert len(requests) == 6
    assert file_hash(resumed.path("images/cue_0001.png")) == first_hash
    assert file_hash(resumed.path("audio/narration.wav")) == voice_hash
    assert resumed.manifest.stages["image.cue_0001"].attempts == 1
    Pipeline(Store(resumed.folder, store.config)).images()
    assert len(requests) == 6


def test_image_model_changes_regenerate_references_and_cues_only(store, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    store.manifest.cues = [Cue(id="cue_0001", text="A scene.", start=0, end=1, word_start=0, word_end=2,
                              scene_group="camp", scene_description="Parent at camp.")]
    calls = []
    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=image_response(png()))
    mock_http(monkeypatch, handler)
    text_settings = store.config.text_cache_settings.copy()
    Pipeline(store).images()
    Pipeline(store).images()
    assert len(calls) == 2
    store.config.openrouter_image_model = "provider/another-image-model"
    Pipeline(store).images()
    assert calls == ["google/gemini-3.1-flash-lite-image"]*2 + ["provider/another-image-model"]*2
    assert store.config.text_cache_settings == text_settings
    assert len(list((store.config.jobs_dir / "references").glob("*.png"))) == 2


def test_image_provider_is_part_of_cache_key():
    config = Config()
    before = config.image_cache_settings
    config.image_provider = "gemini"
    config.gemini_model = config.openrouter_image_model
    assert before != config.image_cache_settings


def test_image_http_400_keeps_redacted_upstream_reason_and_request_id(store, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter-key")
    body = {"error": {"code": 400, "message": "Provider returned error", "metadata": {
        "provider_name": "Google", "raw": json.dumps({"error": {"status": "INVALID_ARGUMENT",
            "message": "Gemini could not generate an image (STOP); credential=fixture-openrouter-key"}})}}}
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(400, json=body, headers={"x-request-id": "request-image-fixture"})
    mock_http(monkeypatch, handler)
    with pytest.raises(PipelineError, match=r"Gemini could not generate an image \(STOP\)") as error:
        Providers(store).image("A scene.", [])
    assert not isinstance(error.value, TemporaryError)
    assert calls == [1]
    entry = store.manifest.usage[-1]
    assert entry["status"] == "failed" and entry["http_status"] == 400
    assert entry["request_id"] == "request-image-fixture"
    saved = read_json(store.path(entry["response_path"]))
    assert saved["http_status"] == 400
    assert saved["error"]["provider_name"] == "Google"
    assert saved["error"]["upstream_status"] == "INVALID_ARGUMENT"
    assert "[REDACTED]" in saved["error"]["upstream_message"]
    assert "fixture-openrouter-key" not in str(error.value)
    assert "fixture-openrouter-key" not in store.path(entry["response_path"]).read_text(encoding="utf-8")
    assert "fixture-openrouter-key" not in store.path("manifest.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("status,body,message", [
    (402, {"detail": {"status": "payment_required", "message": "Free users cannot use library voices via the API"}}, "payment_required"),
    (400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Invalid reference image"}}, "Invalid reference image"),
    (429, {"error": "Slow down"}, "Slow down"),
])
def test_provider_http_errors_include_structured_reason_without_changing_retry_policy(store, monkeypatch, status, body, message):
    mock_http(monkeypatch, lambda request: httpx.Response(status, json=body))
    expected = TemporaryError if status == 429 else PipelineError
    with pytest.raises(expected, match=message):
        Providers(store).post("image", "https://example.org/image", {"Authorization": "Bearer fixture-secret"}, {})
    entry = store.manifest.usage[-1]
    assert entry["status"] == ("temporary_failure" if status == 429 else "failed")
    assert store.path(entry["response_path"]).is_file()


def test_http_error_sanitizes_header_and_environment_secrets_before_truncation(store, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "another-fixture-secret")
    body = {"error": {"message": "Bearer fixture-header-secret another-fixture-secret " + "x"*3000}}
    mock_http(monkeypatch, lambda request: httpx.Response(400, json=body))
    with pytest.raises(PipelineError) as error:
        Providers(store).post("image", "https://example.org/image", {"Authorization": "Bearer fixture-header-secret"}, {})
    saved = store.path(store.manifest.usage[-1]["response_path"]).read_text(encoding="utf-8")
    for secret in ("fixture-header-secret", "another-fixture-secret"):
        assert secret not in str(error.value) and secret not in saved
    assert len(read_json(store.path(store.manifest.usage[-1]["response_path"]))["error"]["message"]) <= 1000


def test_non_json_http_error_saves_status_without_echoing_response_body(store, monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(400, text="<html>private debug body</html>"))
    with pytest.raises(PipelineError, match="HTTP 400") as error:
        Providers(store).post("image", "https://example.org/image", {}, {})
    saved = store.path(store.manifest.usage[-1]["response_path"]).read_text(encoding="utf-8")
    assert "private debug body" not in str(error.value) and "private debug body" not in saved
