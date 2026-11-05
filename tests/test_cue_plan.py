import json
import math

import httpx
import pytest

from test_pipeline import FakeProviders, config, image_bytes, store
from test_timing import timed_words
from video_automation import pipeline as orchestration
from video_automation.models import ScriptPause
from video_automation.pipeline import Pipeline
from video_automation.providers import CUE_SPLIT_SCHEMA, Providers, validate_cue_plan
from video_automation.store import PipelineError, Store, TemporaryError, atomic_bytes, file_hash, read_json
from video_automation.timing import segment_words


def response(data):
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(data)}]}]}


def test_ai_boundaries_keep_arctic_sentence_together_and_preserve_frames():
    first = "In 1845, two British ships sailed toward the Arctic, carrying 129 men and the hopes of an empire."
    second = "None of those men would return."
    words = timed_words(first + " " + second, seconds=0.44)
    duration = words[-1].end + 0.5
    cues = segment_words(words, 30, duration, cue_end_word_indices=[18, 24])
    assert [cue.text for cue in cues] == [first, second]
    assert cues[0].start == words[0].start and cues[0].end == words[17].end
    assert cues[1].start == words[18].start and cues[1].end == words[-1].end
    assert cues[0].start_frame == 0
    assert cues[0].end_frame == cues[1].start_frame == round(words[18].start * 30)
    assert cues[-1].end_frame == math.ceil(duration * 30)


@pytest.mark.parametrize("ends", [[], [0, 4], [-1, 4], [3], [5], [2, 2, 4], [3, 2, 4],
                                      [1.0, 4], [True, 4], ["2", 4], [None, 4]])
def test_invalid_boundaries_are_rejected(ends):
    with pytest.raises(TemporaryError, match="Invalid cue plan"):
        validate_cue_plan({"cue_end_word_indices": ends}, 0, 4)


def test_explicit_pauses_are_mandatory_but_outer_pauses_need_no_empty_cue():
    words = timed_words("One two three four.")
    with pytest.raises(TemporaryError, match="explicit pauses"):
        segment_words(words, 30, 2, [2], cue_end_word_indices=[4])
    cues = segment_words(words, 30, 2, [0, 2, 4], cue_end_word_indices=[2, 4])
    assert [cue.text for cue in cues] == ["One two", "three four."]


def test_ai_boundary_frame_collapse_keeps_all_words():
    words = timed_words("One two three.", seconds=0.001)
    cues = segment_words(words, 30, 1, cue_end_word_indices=[1, 2, 3])
    assert len(cues) == 1 and cues[0].text == "One two three."
    assert (cues[0].start_frame, cues[0].end_frame) == (0, 30)


def test_plan_retries_invalid_result_then_reuses_checkpoint(store):
    words = timed_words("One two. Three four.")
    store.manifest.mode = "script"
    store.manifest.script_pauses = [ScriptPause(word_index=2, seconds=1)]
    provider = FakeProviders()
    received = []

    def cue_plan(context, instructions):
        received.append((context, instructions))
        return {"cue_end_word_indices": [4] if len(received) == 1 else [2, 4]}

    provider.cue_plan = cue_plan
    pipeline = Pipeline(store, provider)
    assert pipeline.cue_plan(words, "One two. Three four.") == [2, 4]
    assert len(received) == 2
    assert received[0][0]["required_pause_boundaries"] == [2]
    assert received[0][0]["words"][0]["text"] == "One"
    assert received[0][1] == orchestration.CUE_SPLIT_PROMPT
    assert store.manifest.stages["cue.plan.000000"].attempts == 2
    assert Pipeline(Store(store.folder, store.config), provider).cue_plan(words, "One two. Three four.") == [2, 4]
    assert len(received) == 2


@pytest.mark.parametrize("change", ["prompt", "model", "alignment", "narration", "cached_data", "file"])
def test_plan_cache_tracks_inputs_and_recovers_corruption(store, monkeypatch, change):
    words = timed_words("One two three four.")
    narration = "One two three four."
    provider = FakeProviders()
    calls = []
    provider.cue_plan = lambda context, instructions: calls.append(context) or {"cue_end_word_indices": [4]}
    pipeline = Pipeline(store, provider)
    atomic_bytes(store.path("audio/narration.wav"), b"preserved speech")
    before = file_hash(store.path("audio/narration.wav"))
    pipeline.cue_plan(words, narration)
    if change == "prompt":
        monkeypatch.setattr(orchestration, "CUE_SPLIT_PROMPT", "My grouping rules.")
    elif change == "model":
        store.config.openrouter_model = "other/model"
    elif change == "alignment":
        words[-1].end += 0.1
    elif change == "narration":
        narration += "\n"
    elif change == "cached_data":
        store.manifest.stages["cue.plan.000000"].result["data"]["cue_end_word_indices"] = [3]
    else:
        atomic_bytes(store.path("cue_plans/words_000000.json"), b"corrupt")
    assert pipeline.cue_plan(words, narration) == [4]
    assert len(calls) == 2
    assert file_hash(store.path("audio/narration.wav")) == before


def test_batched_plan_preserves_completed_section_on_failure(store):
    narration = "One two three four. " * 125
    words = timed_words(narration)
    provider = FakeProviders()
    calls = []
    fail = [True]

    def cue_plan(context, instructions):
        calls.append(context)
        if context["word_start"] == 240 and fail[0]:
            raise TemporaryError("temporary planning failure")
        return {"cue_end_word_indices": [context["word_end"]]}

    provider.cue_plan = cue_plan
    with pytest.raises(TemporaryError):
        Pipeline(store, provider).cue_plan(words, narration)
    checkpoint = file_hash(store.path("cue_plans/words_000000.json"))
    assert store.manifest.stages["cue.plan.000000"].status == "completed"
    fail[0] = False
    store.manifest.stages["cue.plan.000240"].attempts = 0
    store.save()
    assert Pipeline(Store(store.folder, store.config), provider).cue_plan(words, narration) == [240, 480, 500]
    assert [context["word_start"] for context in calls] == [0, 240, 240, 240, 480]
    assert all(context["word_end"] - context["word_start"] <= 240 for context in calls)
    assert calls[1]["words"][0]["index"] == 216
    assert calls[0]["narration"] == narration
    assert file_hash(store.path("cue_plans/words_000000.json")) == checkpoint


@pytest.mark.parametrize("provider", ["openrouter", "openai"])
def test_provider_uses_selected_route_schema_and_editable_instructions(store, monkeypatch, provider):
    store.config.text_provider = provider
    monkeypatch.setenv(store.config.text_key_env, "fixture-key")
    client = Providers(store)
    requests = []

    def post(operation, url, headers, payload):
        requests.append((operation, url, payload))
        return response({"cue_end_word_indices": [4]})

    monkeypatch.setattr(client, "post", post)
    context = {"narration": "One two three four.", "word_start": 0, "word_end": 4,
               "words": [], "required_pause_boundaries": []}
    assert client.cue_plan(context, "My grouping rules.") == {"cue_end_word_indices": [4]}
    operation, url, payload = requests[0]
    assert operation == "visuals" and provider in url
    assert payload["text"]["format"]["schema"] == CUE_SPLIT_SCHEMA
    prompt = payload["input"][1]["content"][0]["text"]
    assert "My grouping rules." in prompt and json.dumps(context) in prompt


@pytest.mark.parametrize("kind", ["malformed", "schema", "boundaries", "refusal"])
def test_provider_failure_policy(store, monkeypatch, kind):
    monkeypatch.setenv(store.config.text_key_env, "fixture-key")
    client = Providers(store)
    data = response({"cue_end_word_indices": [3]})
    if kind == "malformed":
        data["output"][0]["content"][0]["text"] = "not JSON"
    elif kind == "schema":
        data = response({"text": "rewritten narration"})
    elif kind == "refusal":
        data["output"][0]["content"] = [{"type": "refusal"}]
    monkeypatch.setattr(client, "post", lambda *args: data)
    with pytest.raises(PipelineError) as error:
        Pipeline(store, client).cue_plan(timed_words("One two three four."), "One two three four.")
    assert isinstance(error.value, TemporaryError) == (kind != "refusal")
    assert store.manifest.stages["cue.plan.000000"].attempts == (1 if kind == "refusal" else store.config.max_attempts)


def test_plan_reserves_visuals_budget_and_caches_raw_response(store, monkeypatch):
    store.config.spending_ceiling_usd = 0.2
    store.config.request_cost_caps_usd = {"visuals": 0.2}
    monkeypatch.setenv(store.config.text_key_env, "fixture-key")
    original = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response({"cue_end_word_indices": [4]})))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    pipeline = Pipeline(store)
    words = timed_words("One two three four.")
    assert pipeline.cue_plan(words, "One two three four.") == [4]
    assert pipeline.cue_plan(words, "One two three four.") == [4]
    assert len(store.manifest.usage) == 1
    assert store.manifest.usage[0]["operation"] == "visuals"
    assert store.path(store.manifest.usage[0]["response_path"]).is_file()
    with pytest.raises(PipelineError, match="Spending ceiling"):
        pipeline.cue_plan(words, "Changed context.")


def test_full_workflow_uses_ai_boundaries_before_visuals(config, tmp_path, monkeypatch):
    config.image_mode = "external"
    narration = "One two three four. Five six seven eight."
    source = tmp_path / "input.txt"
    source.write_text(narration, encoding="utf-8")
    pipeline = Pipeline.create(config, script_path=source, thumbnail_enabled=False)
    provider = FakeProviders()
    calls = []
    provider.cue_plan = lambda context, instructions: calls.append(context) or {"cue_end_word_indices": [8]}
    original_visuals = provider.visuals

    def visuals(cues, previous_scenes=None, context=None):
        plan = original_visuals(cues, previous_scenes, context)
        for scene, cue in zip(plan["cues"], cues):
            scene["scene_description"] += " Scene depicts: " + cue.text
        return plan

    provider.visuals = visuals
    pipeline.providers = provider
    assert pipeline.run()["status"] == "waiting_for_images"
    assert [cue.text for cue in pipeline.store.manifest.cues] == [narration]
    assert len(read_json(pipeline.store.path("image_prompts.json"))) == 1
    speech_hash = file_hash(pipeline.store.path("audio/narration.wav"))
    resumed = Pipeline(Store(pipeline.store.folder, config), provider)
    assert resumed.run()["status"] == "waiting_for_images"
    assert len(calls) == 1
    assert file_hash(pipeline.store.path("audio/narration.wav")) == speech_hash
    atomic_bytes(resumed.store.path("images/cue_0001.png"), image_bytes())
    assert resumed.run()["passed"]
    assert resumed.store.manifest.status == "completed"
    assert read_json(resumed.store.path("quality_report.json"))["passed"]
    assert len(calls) == 1
    # A changed grouping prompt replans cues and downstream visuals without buying speech again.
    monkeypatch.setattr(orchestration, "CUE_SPLIT_PROMPT", "Keep sentences separate.")
    provider.cue_plan = lambda context, instructions: {"cue_end_word_indices": [4, 8]}
    assert resumed.run()["status"] == "waiting_for_images"
    assert [cue.text for cue in resumed.store.manifest.cues] == ["One two three four.", "Five six seven eight."]
    assert file_hash(resumed.store.path("audio/narration.wav")) == speech_hash
    assert any("different prompt" in record["error"]
               for record in read_json(resumed.store.path("external_images.json"))["invalid_images"])


@pytest.mark.parametrize("sentence_length,expected", [(13, [234, 260]), (300, [240, 300])])
def test_sections_prefer_sentence_end_and_bound_unbroken_sentences(store, sentence_length, expected):
    narration = ((" ".join(["word"] * sentence_length) + ". ") * (20 if sentence_length == 13 else 1)).strip()
    words = timed_words(narration)
    provider = FakeProviders()
    provider.cue_plan = lambda context, instructions: {"cue_end_word_indices": [context["word_end"]]}
    assert Pipeline(store, provider).cue_plan(words, narration) == expected
