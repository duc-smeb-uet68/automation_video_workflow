import json

import httpx
import pytest

from test_pipeline import FakeProviders, narration
from video_automation.cli import main
from video_automation.models import Config
from video_automation.pipeline import Pipeline
from video_automation.store import PipelineError, Store, TemporaryError, atomic_bytes, file_hash, read_json


@pytest.fixture
def pipeline(tmp_path):
    config = Config.load().model_copy(update={"jobs_dir": tmp_path / "jobs", "voice_id": "fixture-voice",
        "width": 640, "height": 360, "max_attempts": 2, "retry_backoff_seconds": 0, "image_interval_seconds": 0})
    source = tmp_path / "narration.txt"
    source.write_bytes(("\ufeff \r\n" + narration() + "\r\n ").encode("utf-8"))
    result = Pipeline.create(config, script_path=source)
    result.providers = FakeProviders()
    return result


def test_metadata_saved_before_speech_and_reused_on_resume(pipeline):
    provider, store = pipeline.providers, pipeline.store
    original = provider.metadata

    def metadata(text):
        assert text == narration()
        return original(text)

    def speech(text):
        provider.calls.append("speech")
        raise PipelineError("speech: quota_exceeded fixture")

    provider.metadata, provider.speech = metadata, speech
    for attempt in range(2):
        run = pipeline if attempt == 0 else Pipeline(Store(store.folder, store.config), provider)
        with pytest.raises(PipelineError, match="quota_exceeded fixture"):
            run.run()
    assert provider.calls == ["metadata", "speech", "speech"]
    assert read_json(store.path("video_metadata.json"))["title"] == "Caregivers Beside the Fire"
    assert store.manifest.stages["metadata"].status == "completed"
    assert store.path("script.txt").read_text(encoding="utf-8") == narration()
    assert read_json(store.path("progress.json"))["phase"] == "speech"


@pytest.mark.parametrize("invalid", [
    {"title": "Title"},
    {"title": " ", "description": "Description"},
    {"title": "Two\nlines", "description": "Description"},
    {"title": "x" * 101, "description": "Description"},
    {"title": "Title", "description": "\t\n "},
    {"title": "Title", "description": "x" * 5001},
    {"title": "Title", "description": "Description", "extra": "Unexpected"},
])
def test_invalid_metadata_never_completes_or_reaches_speech(pipeline, invalid):
    provider = pipeline.providers

    def metadata(text):
        provider.calls.append("metadata")
        return invalid

    provider.metadata = metadata
    with pytest.raises(TemporaryError, match="Invalid video metadata"):
        pipeline.run()
    assert provider.calls == ["metadata"] * pipeline.config.max_attempts
    assert not pipeline.store.path("video_metadata.json").exists()
    assert pipeline.store.manifest.stages["metadata"].status == "failed"
    assert read_json(pipeline.store.path("progress.json"))["phase"] == "metadata"


def test_metadata_recovery_preserves_prepared_speech_and_images(pipeline):
    assert pipeline.run()["status"] == "waiting_for_images"
    store, provider = pipeline.store, pipeline.providers
    preserved = {name: file_hash(store.path(name)) for name in (
        "script.txt", "audio/narration.wav", "alignment.json", "transcript.json", "image_prompts.json", "subtitles.en.srt")}
    # Simulate a compatible job completed before metadata generation existed.
    del store.manifest.stages["metadata"]
    store.path("video_metadata.json").unlink()
    store.save()
    before = list(provider.calls)
    resumed = Pipeline(Store(store.folder, store.config), provider)
    assert resumed.run()["status"] == "waiting_for_images"
    assert provider.calls == before + ["metadata"]
    before = list(provider.calls)
    atomic_bytes(store.path("video_metadata.json"), b"corrupt metadata")
    assert resumed.run()["status"] == "waiting_for_images"
    assert provider.calls == before + ["metadata"]
    assert preserved == {name: file_hash(store.path(name)) for name in preserved}


@pytest.mark.parametrize("change", ["prompt", "model", "narration"])
def test_metadata_cache_tracks_its_own_generation_inputs(pipeline, monkeypatch, change):
    import video_automation.pipeline as orchestration
    script = pipeline.script()
    pipeline.metadata(script)
    pipeline.metadata(script)
    assert pipeline.providers.calls == ["metadata"]
    if change == "prompt":
        monkeypatch.setattr(orchestration, "METADATA_PROMPT", orchestration.METADATA_PROMPT + "\nNew wording.")
    elif change == "model":
        pipeline.config.openrouter_model = "provider/new-model"
    else:
        script.narration = script.narration.replace("fire", "lake")
    pipeline.metadata(script)
    assert pipeline.providers.calls == ["metadata", "metadata"]


def test_invalid_cached_metadata_is_revalidated_before_reuse(pipeline):
    script = pipeline.script()
    pipeline.metadata(script)
    stage = pipeline.store.manifest.stages["metadata"]
    stage.result["data"]["title"] = " "
    pipeline.store.save()
    resumed = Pipeline(Store(pipeline.store.folder, pipeline.config), pipeline.providers)
    assert resumed.metadata(script)["title"] == "Caregivers Beside the Fire"
    assert pipeline.providers.calls == ["metadata", "metadata"]
    assert resumed.store.manifest.stages["metadata"].attempts == 2


@pytest.mark.parametrize("provider,key,endpoint", [
    ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/responses"),
    ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1/responses"),
])
def test_metadata_provider_uses_final_narration_and_strict_schema(pipeline, monkeypatch, provider, key, endpoint):
    store = pipeline.store
    store.config.text_provider = provider
    store.config.spending_ceiling_usd = 0.5
    store.config.request_cost_caps_usd = {"metadata": 0.1}
    monkeypatch.setenv(key, "fixture-key")
    expected = {"title": "Caregivers Beside the Fire", "description": "A grounded summary.\n\nAnother paragraph."}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message",
            "content": [{"type": "output_text", "text": json.dumps(expected)}]}]})

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    assert Pipeline(store).metadata(pipeline.script()) == expected
    assert len(requests) == 1 and str(requests[0].url) == endpoint
    payload = json.loads(requests[0].content)
    prompt = payload["input"][1]["content"][0]["text"]
    assert json.loads(prompt.split("\nNarration: ", 1)[1]) == narration()
    assert "tools" not in payload
    schema = payload["text"]["format"]
    assert schema["strict"] is True and schema["name"] == "metadata"
    assert schema["schema"]["required"] == ["title", "description"]
    assert schema["schema"]["additionalProperties"] is False
    assert store.manifest.usage[0]["operation"] == "metadata"
    assert store.manifest.usage[0]["reserved_upper_bound_usd"] == 0.1
    assert store.path(store.manifest.usage[0]["response_path"]).is_file()


@pytest.mark.parametrize("output,temporary", [
    ({"type": "output_text", "text": "not JSON"}, True),
    ({"type": "output_text", "text": json.dumps({"title": "Title"})}, True),
    ({"type": "output_text", "text": json.dumps({"title": " ", "description": "Summary"})}, True),
    ({"type": "refusal", "refusal": "Fixture refusal"}, False),
])
def test_metadata_provider_retries_malformed_output_but_not_refusals(pipeline, monkeypatch, output, temporary):
    from video_automation.providers import Providers
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    provider = Providers(pipeline.store)
    requests = []

    def post(*args, **kwargs):
        requests.append(1)
        return {"status": "completed", "output": [{"type": "message", "content": [output]}]}

    monkeypatch.setattr(provider, "post", post)
    pipeline.providers = provider
    with pytest.raises(PipelineError) as error:
        pipeline.metadata(pipeline.script())
    assert isinstance(error.value, TemporaryError) == temporary
    assert len(requests) == (pipeline.config.max_attempts if temporary else 1)
    assert not pipeline.store.path("video_metadata.json").exists()


@pytest.mark.parametrize("caps,ceiling,error", [
    ({}, 1, "Missing conservative request_cost_caps_usd for metadata"),
    ({"metadata": 0.2}, 0.1, "Spending ceiling would be exceeded before metadata"),
])
def test_metadata_budget_blocks_requests_before_network(pipeline, monkeypatch, caps, ceiling, error):
    pipeline.config.spending_ceiling_usd = ceiling
    pipeline.config.request_cost_caps_usd = caps
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("Network must not be reached"))
    with pytest.raises(PipelineError, match=error):
        Pipeline(pipeline.store).metadata(pipeline.script())
    assert pipeline.store.manifest.usage == []


def test_cli_reports_metadata_path_at_external_handoff(pipeline, tmp_path, monkeypatch, capsys):
    settings = tmp_path / "config.json"
    settings.write_text(pipeline.config.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr("video_automation.pipeline.Providers", lambda store: pipeline.providers)
    assert main(["--config", str(settings), "resume", pipeline.store.manifest.job_id]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "waiting_for_images"
    assert output["metadata"] == str(pipeline.store.path("video_metadata.json"))
    assert read_json(pipeline.store.path("video_metadata.json"))["description"]
