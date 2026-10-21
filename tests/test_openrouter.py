import json

import httpx
import pytest

from video_automation import media
from video_automation.cli import doctor, load_credentials, main
from video_automation.models import Config, Cue, Manifest
from video_automation.pipeline import Pipeline
from video_automation.providers import Providers, SCRIPT_SCHEMA, VISUAL_SCHEMA
from video_automation.store import PipelineError, Store, TemporaryError, read_json


@pytest.fixture
def store(tmp_path):
    config = Config.load().model_copy(update={"jobs_dir": tmp_path / "jobs", "voice_id": "fixture-voice"})
    return Store(config.jobs_dir / "routing", config, Manifest(job_id="routing", topic="Caregivers", image_strategy="text_only"))


def response(result, annotations=None):
    return {"status": "completed", "output": [{"type": "message", "content": [{
        "type": "output_text", "text": json.dumps(result), "annotations": annotations or []}]}]}


def test_openrouter_default_and_model_selection():
    config = Config()
    assert config.text_provider == "openrouter"
    assert config.text_model == "openai/gpt-oss-120b"
    assert config.generation_keys == ("OPENROUTER_API_KEY", "ELEVENLABS_API_KEY")
    config.openrouter_model = "provider/custom-model"
    assert config.text_model == "provider/custom-model"
    config.text_provider = "openai"
    assert config.text_model == "gpt-4.1" and config.text_key_env == "OPENAI_API_KEY"


@pytest.mark.parametrize("provider,key,endpoint,model", [
    ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/responses", "openai/gpt-oss-120b"),
    ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1/responses", "gpt-4.1"),
])
def test_selected_endpoint_key_and_strict_schema(store, monkeypatch, provider, key, endpoint, model):
    store.config.text_provider = provider
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv(key, "fixture-selected-key")
    client = Providers(store)
    requests = []
    def post(operation, url, headers, payload):
        requests.append((url, headers, payload))
        return response({"narration": "Fixture.", "source_references": []})
    monkeypatch.setattr(client, "post", post)
    result, _ = client.structured("script", "Write a fixture", SCRIPT_SCHEMA)
    url, headers, payload = requests[0]
    assert url == endpoint and headers["Authorization"] == "Bearer fixture-selected-key"
    assert payload["model"] == model and payload["store"] is False
    assert payload["text"]["format"]["strict"] is True
    assert payload["input"][1]["content"][0] == {"type": "input_text", "text": "Write a fixture"}
    assert "tools" not in payload
    if provider == "openrouter":
        assert payload["provider"] == {"require_parameters": True}
    assert result["narration"] == "Fixture."


@pytest.mark.parametrize("nested", [False, True])
def test_openrouter_research_search_limits_and_citation_provenance(store, monkeypatch, nested):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    result = {"summary": "Evidence", "sources": [{"url": "https://example.org/study", "title": "Study", "supported_claims": ["Claim"]}]}
    citation = {"type": "url_citation", "url": "https://example.org/study"}
    if nested:
        citation = {"type": "url_citation", "url_citation": {"url": "https://example.org/study"}}
    requests = []
    def post(operation, url, headers, payload):
        requests.append(payload)
        return response(result, [citation])
    monkeypatch.setattr(client, "post", post)
    research = client.research("Ancient caregivers")
    assert research["retrieved_urls"] == ["https://example.org/study"]
    payload = requests[0]
    assert payload["max_tool_calls"] == 3 and payload["tool_choice"] == "required"
    assert payload["tools"] == [{"type": "openrouter:web_search", "parameters": {
        "engine": "exa", "max_results": 5, "max_total_results": 15, "max_uses": 3, "search_context_size": "medium"}}]
    assert "include" not in payload and "plugins" not in payload
    result["sources"][0]["url"] = "https://example.org/invented"
    with pytest.raises(PipelineError, match="not retrieved"):
        client.research("Ancient caregivers")


def test_openrouter_research_requires_retrieved_citations(store, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    result = {"summary": "Unsupported", "sources": [{"url": "https://example.org/invented", "title": "Study", "supported_claims": ["Claim"]}]}
    monkeypatch.setattr(client, "post", lambda *args: response(result))
    with pytest.raises(PipelineError, match="no verifiable retrieved"):
        client.research("A topic")


@pytest.mark.parametrize("invented", [False, True])
def test_openrouter_research_formats_prose_without_repeating_search(store, monkeypatch, invented):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    citation = {"type": "url_citation", "url": "https://example.org/study", "title": "Study",
                "content": "Observed care was shared among camp members."}
    raw = response(None, [citation])
    raw["output"][0]["content"][0]["text"] = "## Evidence\nCamp members shared childcare."
    raw["output"].insert(0, {"type": "openrouter:web_search", "action": {"sources": [{"url": citation["url"]}]}})
    result = {"summary": "Evidence", "sources": [{"url": "https://example.org/invented" if invented else citation["url"],
              "title": "Study", "supported_claims": ["Camp members shared childcare."]}]}
    requests = []
    def post(operation, url, headers, payload):
        requests.append((operation, url, payload))
        return raw if len(requests) == 1 else response(result, [{"type": "url_citation", "url": result["sources"][0]["url"]}])
    monkeypatch.setattr(client, "post", post)
    if invented:
        with pytest.raises(PipelineError, match="not retrieved"):
            client.research("Caregivers")
    else:
        assert client.research("Caregivers")["retrieved_urls"] == [citation["url"]]
    assert len(requests) == 2
    operation, endpoint, payload = requests[1]
    assert operation == "research" and endpoint == "https://openrouter.ai/api/v1/responses"
    assert payload["model"] == store.config.text_model
    assert "tools" not in payload and "tool_choice" not in payload
    assert payload["text"]["format"]["strict"] is True
    prompt = payload["input"][1]["content"][0]["text"]
    assert citation["content"] in prompt and raw["output"][1]["content"][0]["text"] in prompt.replace("\\n", "\n")
    assert "supported_claims" in prompt and "JSON" in prompt


@pytest.mark.parametrize("formatted,error", [
    (None, "missing or invalid structured output"),
    ({"summary": "Missing sources"}, "structured output violates schema"),
])
def test_openrouter_research_formatting_is_bounded(store, monkeypatch, formatted, error):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    raw = response(None, [{"type": "url_citation", "url": "https://example.org/study"}])
    raw["output"][0]["content"][0]["text"] = "Research prose, not JSON."
    requests = []
    def post(*args):
        requests.append(args)
        return raw if len(requests) == 1 or formatted is None else response(formatted)
    monkeypatch.setattr(client, "post", post)
    with pytest.raises(PipelineError, match=error):
        client.research("Caregivers")
    assert len(requests) == 2


@pytest.mark.parametrize("tool_type", ["web_search_call", "openrouter:web_search"])
def test_research_accepts_search_action_provenance_without_annotations(store, monkeypatch, tool_type):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    result = {"summary": "Evidence", "sources": [{"url": "https://example.org/study", "title": "Study",
              "supported_claims": ["Camp members shared childcare."]}]}
    raw = response(result)
    raw["output"].insert(0, {"type": tool_type, "action": {"sources": [{"url": "https://example.org/study/"}]}})
    client = Providers(store)
    monkeypatch.setattr(client, "post", lambda *args: raw)
    assert client.research("Caregivers")["retrieved_urls"] == ["https://example.org/study"]


@pytest.mark.parametrize("content,error", [
    ({"type": "output_text", "text": "Prose with no retrieved sources."}, "no verifiable retrieved"),
    ({"type": "refusal", "refusal": "Fixture refusal."}, "refused"),
    ({"type": "output_text", "text": ""}, "missing or invalid structured output"),
])
def test_openrouter_research_does_not_reformat_missing_evidence_or_refusals(store, monkeypatch, content, error):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    requests = []
    def post(*args):
        requests.append(args)
        return {"status": "completed", "output": [{"type": "message", "content": [content]}]}
    monkeypatch.setattr(client, "post", post)
    with pytest.raises(PipelineError, match=error):
        client.research("Caregivers")
    assert len(requests) == 1


@pytest.mark.parametrize("ceiling", [0.15, 0.3])
def test_research_formatting_reserves_budget_and_retains_raw_responses(store, monkeypatch, ceiling):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    store.config.spending_ceiling_usd = ceiling
    store.config.request_cost_caps_usd = {"research": 0.1}
    raw = response(None, [{"type": "url_citation", "url": "https://example.org/study"}])
    raw["output"][0]["content"][0]["text"] = "Camp members shared childcare."
    formatted = response({"summary": "Evidence", "sources": [{"url": "https://example.org/study",
                         "title": "Study", "supported_claims": ["Camp members shared childcare."]}]})
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=raw if len(requests) == 1 else formatted)
    original_client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs))
    if ceiling < 0.2:
        with pytest.raises(PipelineError, match="Spending ceiling"):
            Providers(store).research("Caregivers")
        assert len(requests) == len(store.manifest.usage) == 1
    else:
        assert Providers(store).research("Caregivers")["summary"] == "Evidence"
        assert len(requests) == len(store.manifest.usage) == 2
        assert read_json(store.path(store.manifest.usage[1]["response_path"])) == formatted
    assert read_json(store.path(store.manifest.usage[0]["response_path"])) == raw
    assert all(entry["operation"] == "research" and entry["reserved_upper_bound_usd"] == 0.1
               for entry in store.manifest.usage)


@pytest.mark.parametrize("result", [{"narration": "missing sources"}, {"narration": 42, "source_references": []},
                                   {"narration": "text", "source_references": [], "extra": "unexpected"}])
def test_routed_output_must_pass_local_schema(store, monkeypatch, result):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    monkeypatch.setattr(client, "post", lambda *args: response(result))
    with pytest.raises(PipelineError, match="structured output violates schema"):
        client.structured("script", "Fixture", SCRIPT_SCHEMA)


def test_openrouter_incomplete_response_fails_clearly(store, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    monkeypatch.setattr(client, "post", lambda *args: {"status": "incomplete", "output": []})
    with pytest.raises(PipelineError, match="openrouter response is incomplete"):
        client.structured("script", "Fixture", SCRIPT_SCHEMA)


@pytest.mark.parametrize("provider,selected,unused", [
    ("openrouter", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    ("openai", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
])
def test_doctor_requires_only_selected_text_key(store, monkeypatch, capsys, provider, selected, unused):
    store.config.text_provider = provider
    store.config.image_provider = "gemini"
    monkeypatch.setattr(media, "run", lambda command: "version")
    monkeypatch.delenv(unused, raising=False)
    for name in store.config.generation_keys:
        monkeypatch.setenv(name, "fixture-secret")
    assert doctor(store.config) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["text_provider"] == provider and report["text_model"] == store.config.text_model
    assert report["checks"][selected] and unused not in report["checks"]
    assert "fixture-secret" not in json.dumps(report)


def test_cli_openrouter_preflight_does_not_require_openai_key(store, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in store.config.generation_keys:
        monkeypatch.setenv(name, "fixture-key")
    config_path = tmp_path / "config.json"
    config_path.write_text(store.config.model_dump_json(), encoding="utf-8")
    calls = []
    monkeypatch.setattr(Pipeline, "run", lambda pipeline: calls.append(pipeline.store.manifest.topic))
    assert main(["--config", str(config_path), "run", "--mode", "topic", "Preflight fixture", "--job-id", "preflight"]) == 0
    assert calls == ["Preflight fixture"]


def test_cli_openai_key_cannot_substitute_for_openrouter(store, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fixture-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-key")
    path = tmp_path / "config.json"
    path.write_text(store.config.model_dump_json(), encoding="utf-8")
    assert main(["--config", str(path), "run", "--mode", "topic", "Missing routed key", "--job-id", "missing"]) == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert read_json(store.config.jobs_dir / "missing/manifest.json")["usage"] == []


def test_env_loads_openrouter_key_without_overwriting_existing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text('OPENROUTER_API_KEY="local-fixture"\n', encoding="utf-8")
    load_credentials(path)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "local-fixture"
    monkeypatch.setenv("OPENROUTER_API_KEY", "existing-fixture")
    load_credentials(path)
    assert os.environ["OPENROUTER_API_KEY"] == "existing-fixture"


def test_text_provider_and_model_changes_invalidate_generation_cache(store):
    calls = []
    def action():
        calls.append(1)
        return {"files": []}
    store.task("script", store.config.text_cache_settings, action)
    store.task("script", store.config.text_cache_settings, action)
    store.config.openrouter_model = "openai/gpt-4.1"
    store.task("script", store.config.text_cache_settings, action)
    # Same model string at a different provider must not share generation results.
    store.config.openai_model = store.config.openrouter_model
    store.config.text_provider = "openai"
    store.task("script", store.config.text_cache_settings, action)
    assert len(calls) == 3


@pytest.fixture
def visual_cues():
    return [Cue(id=f"cue_{index:04}", text="A scene.", start=index-25, end=index-24,
                word_start=index-25, word_end=index-24) for index in (25, 26)]


@pytest.fixture
def visual_plan(visual_cues):
    return {"cues": [{"id": cue.id, "scene_group": "camp", "character_ids": ["parent"],
                      "scene_description": "A parent by the fire.", "overlay_text": None, "motion": "hold"}
                     for cue in visual_cues]}


@pytest.mark.parametrize("provider,key", [("openrouter", "OPENROUTER_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_visual_request_schema_constrains_batch_without_mutating_shared_schema(store, monkeypatch, visual_cues, visual_plan, provider, key):
    store.config.text_provider = provider
    monkeypatch.setenv(key, "fixture-key")
    client = Providers(store)
    before = json.dumps(VISUAL_SCHEMA, sort_keys=True)
    requests = []
    def post(operation, url, headers, payload):
        requests.append(payload)
        return response(visual_plan)
    monkeypatch.setattr(client, "post", post)
    assert client.visuals(visual_cues) == visual_plan
    schema = requests[0]["text"]["format"]["schema"]["properties"]["cues"]
    assert schema["minItems"] == schema["maxItems"] == 2
    assert schema["items"]["properties"]["id"]["enum"] == ["cue_0025", "cue_0026"]
    assert "enum" not in schema["items"]["properties"]["scene_description"]
    assert "enum" not in schema["items"]["properties"]["character_ids"]["items"]
    assert json.dumps(VISUAL_SCHEMA, sort_keys=True) == before


@pytest.mark.parametrize("provider", ["openrouter", "openai"])
def test_visual_planner_accepts_narration_specific_characters(store, monkeypatch, visual_cues, visual_plan, provider):
    store.config.text_provider = provider
    monkeypatch.setenv(store.config.text_key_env, "fixture-key")
    visual_plan["cues"][0]["character_ids"] = ["astronaut", "scientist"]
    visual_plan["cues"][1]["character_ids"] = []
    client = Providers(store)
    monkeypatch.setattr(client, "post", lambda *args: response(visual_plan))
    assert client.visuals(visual_cues) == visual_plan
    assert visual_cues[0].character_ids == []


@pytest.mark.parametrize("invalid", ["count", "id", "duplicate", "reordered", "character", "blank", "json"])
def test_invalid_visual_provider_output_is_retryable(store, monkeypatch, visual_cues, visual_plan, invalid):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    if invalid == "count":
        visual_plan["cues"].pop()
    elif invalid == "id":
        visual_plan["cues"][0]["id"] = "..."
    elif invalid == "duplicate":
        visual_plan["cues"][1]["id"] = "cue_0025"
    elif invalid == "reordered":
        visual_plan["cues"].reverse()
    elif invalid == "character":
        visual_plan["cues"][0]["character_ids"] = [" "]
    elif invalid == "blank":
        visual_plan["cues"][0]["scene_group"] = " "
    output = response(visual_plan)
    if invalid == "json":
        output["output"][0]["content"][0]["text"] = "not JSON"
    client = Providers(store)
    monkeypatch.setattr(client, "post", lambda *args: output)
    with pytest.raises(TemporaryError):
        client.visuals(visual_cues)


def test_visual_refusal_remains_non_retryable(store, monkeypatch, visual_cues):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    client = Providers(store)
    output = {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}
    monkeypatch.setattr(client, "post", lambda *args: output)
    with pytest.raises(PipelineError, match="refused") as error:
        client.visuals(visual_cues)
    assert not isinstance(error.value, TemporaryError)
