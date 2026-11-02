import json
from copy import deepcopy

import httpx
import pytest

from test_pipeline import FakeProviders, config, store, visual_words
from video_automation.models import Cue
from video_automation.pipeline import Pipeline
from video_automation.providers import Providers, STYLE, VISUAL_CONTEXT_SCHEMA, validate_visual_context
from video_automation.store import PipelineError, Store, TemporaryError, atomic_bytes, file_hash, read_json


def context():
    return {"summary": "A caregiver explains why the child needs warmth before returning to the fire.",
            "characters": [{"id": "parent", "description": "A round white-headed parent in a blue tunic."}]}


def response(data):
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(data)}]}]}


@pytest.mark.parametrize("provider", ["openrouter", "openai"])
def test_shared_context_and_style_reach_both_text_providers(store, monkeypatch, provider):
    store.config.text_provider = provider
    monkeypatch.setenv(store.config.text_key_env, "fixture-key")
    requests = []
    client = Providers(store)

    def post(operation, url, headers, payload):
        requests.append((operation, payload))
        if "summary" in payload["text"]["format"]["schema"]["properties"]:
            return response(context())
        return response({"cues": {"cue_0001": {"scene_group": "camp", "character_ids": ["parent"],
            "scene_description": context()["characters"][0]["description"] + " Standing by a fire.", "motion": "hold"}}})

    monkeypatch.setattr(client, "post", post)
    narration = "A child needs warmth.\nLater, return to the fire."
    shared = client.visual_context(narration)
    cues = [Cue(id="cue_0001", text="A child needs warmth.", start=0, end=1, word_start=0, word_end=4)]
    client.visuals(cues, [], shared)
    first, batch = [request[1]["input"][1]["content"][0]["text"] for request in requests]
    assert json.dumps(narration, ensure_ascii=False) in first
    assert requests[0][1]["text"]["format"]["schema"] == VISUAL_CONTEXT_SCHEMA
    assert all(operation == "visuals" for operation, _ in requests)
    assert STYLE in first and STYLE in batch
    assert json.dumps(shared, ensure_ascii=False) in batch
    assert "one main idea per scene" in batch and "2–5 words" in batch
    assert "Repeat the supplied character descriptions in full" in batch


def test_script_workflow_plans_from_exact_finalized_narration(config, tmp_path):
    config.image_mode = "external"
    narration = 'A child needs warmth.\n\nThe label is "38%"; this is an illustration.'
    source = tmp_path / "narration.txt"
    source.write_text(narration, encoding="utf-8")
    pipeline = Pipeline.create(config, script_path=source, thumbnail_enabled=False)
    provider = FakeProviders()
    original = provider.visual_context
    received = []

    def visual_context(text):
        received.append(text)
        return original(text)

    provider.visual_context = visual_context
    pipeline.providers = provider
    assert pipeline.run()["status"] == "waiting_for_images"
    assert received == [narration]
    assert provider.spoken == narration
    assert "image" not in provider.calls
    before = list(provider.calls)
    assert Pipeline(Store(pipeline.store.folder, config), provider).run()["status"] == "waiting_for_images"
    assert provider.calls == before


@pytest.mark.parametrize("invalid", [
    {}, {"summary": " ", "characters": []}, {"summary": "Summary", "characters": "parent"},
    {"summary": "Summary", "characters": [{"id": "parent", "description": " "}]},
    {"summary": "Summary", "characters": [{"id": " parent", "description": "Blue tunic"}]},
    {"summary": "Summary", "characters": [{"id": "parent", "description": "Blue tunic"}] * 2},
    {"summary": "Summary", "characters": [], "extra": True},
])
def test_invalid_shared_context_is_rejected(invalid):
    with pytest.raises(TemporaryError, match="Invalid visual context"):
        validate_visual_context(invalid)


def test_shared_context_survives_batch_boundaries_and_cached_resume(store):
    provider = FakeProviders()
    provider.visual_context = lambda narration: context()
    original = provider.visuals
    received = []

    def visuals(cues, previous_scenes=None, context=None):
        received.append((cues[0].id, deepcopy(context), deepcopy(previous_scenes)))
        return original(cues, previous_scenes, context)

    provider.visuals = visuals
    words = visual_words(store)
    pipeline = Pipeline(store, provider)
    pipeline.visuals(words, "Complete finalized narration, including the ending.")
    assert len(received) == 3
    assert all(shared == context() for _, shared, _ in received)
    assert [len(previous) for _, _, previous in received] == [0, 3, 3]
    assert read_json(store.path("visual_context.json")) == context()
    before = {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns)
              for name in ("visual_context.json", "visuals/batch_000.json", "visuals/batch_001.json")}
    Pipeline(Store(store.folder, store.config), provider).visuals(words, "Complete finalized narration, including the ending.")
    assert len(received) == 3
    assert before == {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns) for name in before}


@pytest.mark.parametrize("change", ["narration", "model", "style", "prompt", "schema"])
def test_context_cache_tracks_generation_inputs(store, monkeypatch, change):
    import video_automation.pipeline as orchestration
    provider = FakeProviders()
    pipeline = Pipeline(store, provider)
    narration = "Complete narration."
    pipeline.visual_context(narration)
    pipeline.visual_context(narration)
    assert provider.calls == ["visual_context"]
    if change == "narration":
        narration += " A different ending."
    elif change == "model":
        pipeline.config.openrouter_model = "provider/new-model"
    elif change == "style":
        monkeypatch.setattr(orchestration, "STYLE", STYLE + " New style instructions.")
    elif change == "prompt":
        monkeypatch.setattr(orchestration, "VISUAL_CONTEXT_PROMPT", orchestration.VISUAL_CONTEXT_PROMPT + " New instructions.")
    else:
        schema = deepcopy(VISUAL_CONTEXT_SCHEMA)
        schema["properties"]["summary"]["minLength"] = 2
        monkeypatch.setattr(orchestration, "VISUAL_CONTEXT_SCHEMA", schema)
    pipeline.visual_context(narration)
    assert provider.calls == ["visual_context", "visual_context"]


def test_changed_shared_appearance_invalidates_batches_but_preserves_speech(store):
    provider = FakeProviders()
    shared = context()
    provider.visual_context = lambda narration: deepcopy(shared)
    words = visual_words(store)
    atomic_bytes(store.path("audio/narration.wav"), b"existing speech")
    preserved = file_hash(store.path("audio/narration.wav"))
    pipeline = Pipeline(store, provider)
    pipeline.visuals(words)
    before = store.manifest.stages["visuals.000"].input_hash
    shared["characters"][0]["description"] = "A round white-headed parent in a green tunic."
    store.path("visual_context.json").unlink()
    pipeline.visuals(words)
    assert provider.calls.count("visuals") == 6
    assert store.manifest.stages["visuals.000"].input_hash != before
    assert file_hash(store.path("audio/narration.wav")) == preserved


@pytest.mark.parametrize("corruption", ["file", "cached_data"])
def test_corrupt_context_recovers_without_regenerating_unchanged_batches(store, corruption):
    provider = FakeProviders()
    words = visual_words(store, 8)
    pipeline = Pipeline(store, provider)
    pipeline.visuals(words)
    if corruption == "file":
        atomic_bytes(store.path("visual_context.json"), b"corrupt")
    else:
        store.manifest.stages["visual.context"].result["data"]["summary"] = " "
        store.save()
    resumed = Pipeline(Store(store.folder, store.config), provider)
    resumed.visuals(words)
    assert provider.calls == ["visual_context", "visuals", "visual_context"]
    assert read_json(store.path("visual_context.json"))["summary"].strip()


def test_unknown_character_retries_using_same_shared_context(store):
    provider = FakeProviders()
    original = provider.visuals

    def visuals(cues, previous_scenes=None, context=None):
        planned = original(cues, previous_scenes, context)
        if provider.calls.count("visuals") == 1:
            planned["cues"][0]["character_ids"] = ["new_unplanned_character"]
        return planned

    provider.visuals = visuals
    Pipeline(store, provider).visuals(visual_words(store, 8))
    assert provider.calls == ["visual_context", "visuals", "visuals"]


@pytest.mark.parametrize("output", ["malformed", "blank", "refusal"])
def test_context_failure_preserves_retry_and_refusal_rules(store, monkeypatch, output):
    client = Providers(store)
    monkeypatch.setenv(store.config.text_key_env, "fixture-key")
    data = response({"summary": " ", "characters": []})
    if output == "malformed":
        data["output"][0]["content"][0]["text"] = "invalid JSON"
    elif output == "refusal":
        data["output"][0]["content"] = [{"type": "refusal"}]
    monkeypatch.setattr(client, "post", lambda *args: data)
    with pytest.raises(PipelineError) as error:
        Pipeline(store, client).visual_context("Final narration.")
    assert isinstance(error.value, TemporaryError) == (output != "refusal")
    assert store.manifest.stages["visual.context"].attempts == (1 if output == "refusal" else store.config.max_attempts)
    assert not store.path("visual_context.json").exists()


def test_context_request_uses_existing_visuals_budget_and_preserves_raw_response(store, monkeypatch):
    store.config.spending_ceiling_usd = 0.2
    store.config.request_cost_caps_usd = {"visuals": 0.2}
    monkeypatch.setenv(store.config.text_key_env, "fixture-key")
    original_client = httpx.Client
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response(context()))

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    pipeline = Pipeline(store)
    assert pipeline.visual_context("Full narration.") == context()
    assert pipeline.visual_context("Full narration.") == context()
    entry = store.manifest.usage[0]
    assert entry["operation"] == "visuals" and entry["reserved_upper_bound_usd"] == 0.2
    assert store.path(entry["response_path"]).is_file()
    with pytest.raises(PipelineError, match="Spending ceiling"):
        pipeline.visual_context("Changed narration.")
    assert len(requests) == len(store.manifest.usage) == 1
