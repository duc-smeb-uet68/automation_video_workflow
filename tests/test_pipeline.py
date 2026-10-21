import base64
import io
import json
import math
import struct
import wave
from pathlib import Path

import httpx
import pytest
from PIL import Image

from video_automation import assets, media
from video_automation.models import Config, Cue, Manifest, Word
from video_automation.pipeline import Pipeline
from video_automation.providers import Providers
from video_automation.store import PipelineError, Store, TemporaryError, atomic_bytes, file_hash, read_json, worker_lock
from video_automation.timing import aligned_words, assign_frames, make_script, normalize, segment_words


@pytest.fixture
def config(tmp_path):
    return Config.load().model_copy(update={"jobs_dir": tmp_path / "jobs", "width": 640, "height": 360,
                                           "image_interval_seconds": 0, "retry_backoff_seconds": 0, "max_attempts": 2})


@pytest.fixture
def store(config):
    return Store(config.jobs_dir / "test", config, Manifest(job_id="test", topic="Ancient caregivers"))


def narration():
    sentence = "You imagine a parent near the fire, while another caregiver watches the child. "
    return normalize(sentence * 62)


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (640, 360), "#fffaf0").save(buffer, "PNG")
    return buffer.getvalue()


class FakeProviders:
    """Offline contract fixture, never substituted into the production CLI."""
    def __init__(self, text=None):
        self.text = text or narration()
        self.calls = []
        self.spoken = None

    def research(self, topic):
        self.calls.append("research")
        return {"sources": [{"url": "https://example.org/study", "title": "Fixture", "supported_claims": ["Fixture"]}]}

    def script(self, topic, research, previous=None):
        self.calls.append("revision" if previous else "script")
        return {"narration": self.text, "source_references": ["https://example.org/study"]}

    def audit(self, script, research):
        self.calls.append("audit")
        return {"sentences": [{"sentence_id": s.id, "supported": True, "kind": "reconstruction",
                               "source_urls": [], "reason": "Explicit reconstruction fixture"} for s in script.sentences]}

    def speech(self, text):
        self.calls.append("speech")
        self.spoken = text
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(48000)
            audio.writeframes(b"".join(struct.pack("<h", int(3000*math.sin(2*math.pi*220*i/48000))) for i in range(48000*8)))
        scale = 7.9/len(text)
        return {"audio_base64": base64.b64encode(buffer.getvalue()).decode(), "alignment": {
            "characters": list(text), "character_start_times_seconds": [i*scale for i in range(len(text))],
            "character_end_times_seconds": [(i+1)*scale for i in range(len(text))]}}

    def visuals(self, cues, previous_scenes=None):
        self.calls.append("visuals")
        return {"cues": [{"id": c.id, "scene_group": "camp", "character_ids": ["parent"],
                          "scene_description": "Parent beside a fire.", "overlay_text": None, "motion": "hold"} for c in cues]}

    def image(self, prompt, references):
        self.calls.append("image")
        return image_bytes()


def test_normalized_unicode_count_and_boundaries():
    assert normalize(" \r\nA\rB\r\n ") == "A\nB"
    for length in (4500, 5000):
        text = "é" * (length-1) + "."
        assert make_script(text).character_count == length
    for length in (4499, 5001):
        with pytest.raises(PipelineError, match="before speech generation"):
            make_script("x" * length)
    script = make_script("You hear a cry.\n\nYou imagine a camp!", enforce_length=False)
    assert [s.id for s in script.sentences] == ["sentence_0001", "sentence_0002"]
    assert [s.paragraph_id for s in script.sentences] == ["paragraph_001", "paragraph_002"]
    assert all(script.narration[s.char_start:s.char_end] == s.text for s in script.sentences)


def test_length_revision_once_blocks_paid_downstream(store):
    provider = FakeProviders("Too short.")
    pipeline = Pipeline(store, provider)
    with pytest.raises(PipelineError, match="required 4500"):
        pipeline.run()
    assert provider.calls == ["research", "script", "revision"]
    assert store.manifest.status == "failed"
    assert store.path("failure_report.json").is_file()
    # Restart does not buy fresh drafts or a third revision.
    with pytest.raises(PipelineError, match="required 4500"):
        Pipeline(Store(store.folder, store.config), provider).run()
    assert provider.calls == ["research", "script", "revision"]


def test_single_length_revision_can_succeed(store):
    provider = FakeProviders()
    original = provider.script
    def repair(topic, research, previous=None):
        if previous is None:
            provider.calls.append("script")
            return {"narration": "Too short.", "source_references": ["https://example.org/study"]}
        return original(topic, research, previous)
    provider.script = repair
    script = Pipeline(store, provider).script()
    assert script.character_count == len(narration())
    assert provider.calls.count("revision") == 1


def test_alignment_requires_same_text_and_valid_timestamps():
    script = make_script("You hear.", enforce_length=False)
    alignment = {"characters": list(script.narration), "character_start_times_seconds": [i/10 for i in range(9)],
                 "character_end_times_seconds": [(i+1)/10 for i in range(9)]}
    words = aligned_words(script, alignment, 1)
    assert [w.text for w in words] == ["You", "hear."]
    alignment["characters"][0] = "I"
    with pytest.raises(PipelineError, match="exact normalized narration"):
        aligned_words(script, alignment, 1)
    alignment["characters"][0] = "Y"
    alignment["character_end_times_seconds"][0] = float("nan")
    with pytest.raises(PipelineError, match="invalid or unordered"):
        aligned_words(script, alignment, 1)


def test_cue_coverage_rounding_and_final_hold():
    words = [Word(text=f"word{i}.", start=i*0.001, end=i*0.001+0.0009,
                  char_start=i*7, char_end=i*7+6, sentence_id=f"s{i}") for i in range(50)]
    cues = segment_words(words, 30, 2.015)
    assert " ".join(c.text for c in cues) == " ".join(w.text for w in words)
    assert cues[0].start_frame == 0
    assert cues[-1].end_frame == math.ceil(2.015*30)
    assert all(c.end_frame > c.start_frame for c in cues)
    assert all(a.end_frame == b.start_frame for a, b in zip(cues, cues[1:]))
    assert cues[-1].word_end == len(words)


def test_task_retries_and_corruption_recovery(store):
    calls = []
    def generate():
        calls.append(1)
        if len(calls) == 1:
            raise TemporaryError("HTTP 429")
        atomic_bytes(store.path("artifact.txt"), b"good")
        return {"files": ["artifact.txt"]}
    store.task("image.test", {"prompt": 1}, generate)
    assert len(calls) == 2
    Store(store.folder, store.config).task("image.test", {"prompt": 1}, generate)
    assert len(calls) == 2
    atomic_bytes(store.path("artifact.txt"), b"corrupt")
    store.task("image.test", {"prompt": 1}, generate)
    assert len(calls) == 3
    assert store.path("artifact.txt").read_bytes() == b"good"


def test_fixed_retry_limit_persists_after_restart(store):
    calls = []
    def fail():
        calls.append(1)
        raise TemporaryError("quota")
    with pytest.raises(TemporaryError):
        store.task("image", {}, fail)
    with pytest.raises(PipelineError, match="retry limit reached"):
        Store(store.folder, store.config).task("image", {}, fail)
    assert len(calls) == store.config.max_attempts


def test_atomic_write_preserves_last_checkpoint_on_replace_failure(store, monkeypatch):
    import video_automation.store as storage
    before = store.path("manifest.json").read_bytes()
    def interrupted_replace(*args):
        raise OSError("simulated failed replace")
    monkeypatch.setattr(storage.os, "replace", interrupted_replace)
    store.manifest.status = "running"
    with pytest.raises(OSError):
        store.save()
    assert store.path("manifest.json").read_bytes() == before
    assert not list(store.folder.glob(".*.tmp"))


def test_budget_rejects_before_network_and_records_usage(store, monkeypatch):
    store.config.spending_ceiling_usd = 0.5
    store.config.request_cost_caps_usd = {"image": 0.4}
    store.reserve_request("image")
    with pytest.raises(PipelineError, match="Spending ceiling"):
        store.reserve_request("image")
    with pytest.raises(PipelineError, match="Missing conservative"):
        store.reserve_request("speech")
    assert len(read_json(store.path("manifest.json"))["usage"]) == 1


def test_provider_http_error_redacts_credentials(store, monkeypatch):
    provider = Providers(store)
    response = httpx.Response(429, json={"error": "secret-value"})
    class Client:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return response
    monkeypatch.setattr(httpx, "Client", Client)
    with pytest.raises(TemporaryError, match="temporary HTTP 429") as error:
        provider.post("image", "https://example.org", {"secret": "secret-value"}, {})
    assert "secret-value" not in str(error.value)
    assert store.manifest.usage[-1]["status"] == "temporary_failure"


def test_provider_structured_research_is_bounded_and_checks_sources(store, monkeypatch):
    store.config.text_provider = "openai"
    provider = Providers(store)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    requests = []
    result = {"summary": "Evidence", "sources": [{"url": "https://example.org/study", "title": "Study", "supported_claims": ["Claim"]}]}
    def post(operation, url, headers, payload):
        requests.append(payload)
        return {"status": "completed", "output": [
            {"type": "web_search_call", "action": {"sources": [{"url": "https://example.org/study"}]}},
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(result)}]}]}
    monkeypatch.setattr(provider, "post", post)
    assert provider.research("Caregivers")["sources"] == result["sources"]
    assert requests[0]["max_tool_calls"] == 3
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["text"]["format"]["strict"] is True
    result["sources"][0]["url"] = "https://example.org/invented"
    with pytest.raises(PipelineError, match="not retrieved"):
        provider.research("Caregivers")


def test_invalid_generated_image_retries_before_approval(store):
    provider = FakeProviders()
    original = provider.image
    calls = []
    def image(prompt, references):
        calls.append(1)
        return b"invalid-image" if len(calls) == 1 else original(prompt, references)
    provider.image = image
    store.manifest.cues = [Cue(id="cue_0001", text="Scene.", start=0, end=1, word_start=0, word_end=1,
                              scene_group="camp", scene_description="Parent by fire.")]
    Pipeline(store, provider).images()
    assert store.manifest.stages["image.character_reference"].attempts == 2
    assert store.manifest.stages["image.cue_0001"].attempts == 1
    assert media.check_image(store.path("images/cue_0001.png"))["width"] == 640


def test_interrupted_image_preserves_previous_images_and_speech(store):
    provider = FakeProviders()
    pipeline = Pipeline(store, provider)
    atomic_bytes(store.path("audio/narration.wav"), b"unchanged-voiceover")
    voice_hash = file_hash(store.path("audio/narration.wav"))
    store.manifest.cues = [Cue(id=f"cue_{i:04}", text="A scene.", start=i-1, end=i,
                             word_start=(i-1)*2, word_end=i*2, scene_group="camp", scene_description=f"scene {i}") for i in range(1, 4)]
    original = provider.image
    def interrupt(prompt, references):
        if "scene 2" in prompt:
            raise KeyboardInterrupt
        return original(prompt, references)
    provider.image = interrupt
    with pytest.raises(KeyboardInterrupt):
        pipeline.images()
    first_hash = file_hash(store.path("images/cue_0001.png"))
    successful_calls = len(provider.calls)
    provider.image = original
    resumed = Pipeline(Store(store.folder, store.config), provider)
    resumed.images()
    assert len(provider.calls) == successful_calls+2
    assert file_hash(store.path("images/cue_0001.png")) == first_hash
    assert file_hash(store.path("audio/narration.wav")) == voice_hash
    assert resumed.store.manifest.stages["image.cue_0001"].attempts == 1
    assert all(resumed.store.path(c.image_path).is_file() for c in resumed.store.manifest.cues)


def test_source_audit_blocks_unsupported_claims(store):
    provider = FakeProviders()
    original = provider.audit
    def unsupported(script, research):
        result = original(script, research)
        result["sentences"][0].update(supported=False, reason="Unsupported fixture")
        return result
    provider.audit = unsupported
    with pytest.raises(PipelineError, match="Unsupported narration sentence"):
        Pipeline(store, provider).run()
    assert "speech" not in provider.calls


def test_production_orchestration_and_resume_with_offline_contracts(store, monkeypatch):
    provider = FakeProviders("\r\n" + narration() + "\r\n")
    monkeypatch.setattr(media, "render", lambda store: None)
    monkeypatch.setattr(media, "finish", lambda store: None)
    monkeypatch.setattr(media, "validate", lambda store: {"passed": True})
    pipeline = Pipeline(store, provider)
    pipeline.run()
    assert provider.spoken == normalize(provider.text)
    assert store.path("script.txt").read_text(encoding="utf-8") == provider.spoken
    assert 4500 <= store.manifest.script_character_count <= 5000
    before = list(provider.calls)
    Pipeline(Store(store.folder, store.config), provider).run()
    assert provider.calls == before
    assert store.manifest.status == "completed"
    assert len(store.manifest.cues) != 81


@pytest.mark.parametrize("mode", ["production", "script"])
def test_offline_full_pipeline_renders_real_mp4(store, tmp_path, mode):
    provider = FakeProviders()
    if mode == "script":
        path = tmp_path / "narration.txt"
        path.write_text(narration(), encoding="utf-8")
        store = Pipeline.create(store.config, script_path=path).store
    pipeline = Pipeline(store, provider)
    report = pipeline.run()
    assert report["passed"] and report["details"]["narration_length"]["enforced"]
    assert 4500 <= store.manifest.script_character_count <= 5000
    assert len(store.manifest.cues) == len(list(store.path("images").glob("cue_*.png")))
    assert store.path("final.mp4").is_file()
    assert store.path("subtitles.en.srt").is_file()
    assert store.manifest.total_frames == 240
    assert report["mode"] == mode
    if mode == "script":
        assert report["details"]["source_audit"] == "Skipped: narration supplied by the user."
        assert not {"research", "script", "revision", "audit"} & set(provider.calls)
        assert not store.path("sources.json").exists()
        assert not store.path("source_audit.json").exists()
        # Script mode still requires fresh speech alignment and the production length gate.
        store.path("alignment.json").unlink()
        with pytest.raises(PipelineError, match="alignment_audio_match"):
            media.validate(store)
        atomic_bytes(store.path("script.txt"), b"Too short.")
        with pytest.raises(PipelineError, match="narration_length"):
            media.validate(store)


def test_renderer_integration_music_and_cache(config):
    pipeline = assets.fixture(config, "integration")
    music = config.jobs_dir / "music.wav"
    media.ffmpeg(config, "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", music)
    config.music_path = music
    report = pipeline.run()
    assert report["passed"] and all(report["checks"].values())
    assert report["details"]["narration_length"]["enforced"] is False
    before = file_hash(pipeline.store.path("final.mp4"))
    Pipeline(Store(pipeline.store.folder, config)).run()
    assert file_hash(pipeline.store.path("final.mp4")) == before
    assert all(s.attempts == 1 for s in pipeline.store.manifest.stages.values())
    # Corrupting subtitles cannot silently pass validation.
    atomic_bytes(pipeline.store.path("subtitles.en.srt"), b"broken")
    with pytest.raises(PipelineError, match="caption_bounds"):
        media.validate(pipeline.store)


def test_worker_lock_and_path_safety(store):
    with worker_lock(store.config.jobs_dir):
        with pytest.raises(PipelineError, match="Another worker"):
            with worker_lock(store.config.jobs_dir):
                pass
    with pytest.raises(PipelineError, match="escapes job directory"):
        store.path("../../outside.txt")


def test_kdenlive_export_uses_dynamic_manifest(config, tmp_path):
    from video_automation.kdenlive import export, properties
    import xml.etree.ElementTree as ET
    pipeline = assets.fixture(config, "editor")
    template = tmp_path / "template.kdenlive"
    template.write_text('''<mlt><profile/>
      <producer id="black"><property name="resource">black</property></producer>
      <producer id="old"><property name="mlt_service">qimage</property></producer>
      <chain id="audio"><property name="mlt_service">avformat</property></chain>
      <playlist id="playlist2"/><playlist id="playlist4"/><playlist id="main_bin"/>
      <tractor id="sequence"><property name="kdenlive:uuid">fixture</property></tractor>
      <tractor id="project"><property name="kdenlive:projectTractor">1</property><track/></tractor>
      </mlt>''', encoding="utf-8")
    result = export(pipeline.store, template)
    root = ET.parse(result).getroot()
    images = [p for p in root.findall("producer") if properties(p).get("mlt_service") == "qimage"]
    assert len(images) == 3
    assert root.find("profile").get("frame_rate_num") == "30"
    chain = root.find("chain")
    assert properties(chain)["resource"] == pipeline.store.path("audio/narration.wav").as_posix()
    assert all(Path(properties(p)["resource"]).is_file() for p in images)


def test_env_file_only_reads_provider_keys(tmp_path, monkeypatch):
    from video_automation.cli import load_credentials
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "existing")
    env = tmp_path / ".env"
    env.write_text('OPENAI_API_KEY="local-value"\nGEMINI_API_KEY=replacement\nUNRELATED_SECRET=ignored\n', encoding="utf-8")
    load_credentials(env)
    import os
    assert os.environ["OPENAI_API_KEY"] == "local-value"
    assert os.environ["GEMINI_API_KEY"] == "existing"
    assert "UNRELATED_SECRET" not in os.environ


def test_cli_missing_setup_fails_before_requests(config, tmp_path, monkeypatch):
    from video_automation.cli import main
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = tmp_path / "config.json"
    settings.write_text(config.model_dump_json(), encoding="utf-8")
    assert main(["--config", str(settings), "run", "--mode", "topic", "Missing setup fixture", "--job-id", "missing_setup"]) == 1
    manifest = read_json(config.jobs_dir / "missing_setup/manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["usage"] == [] and manifest["stages"] == {}
    assert (config.jobs_dir / "missing_setup/failure_report.json").is_file()


def test_overlay_rejects_unfit_text(config, tmp_path):
    with pytest.raises(PipelineError, match="does not fit"):
        media.overlay(config, "x"*1000, tmp_path/"bad.png")


def test_script_mode_normalizes_skips_research_and_resumes_from_snapshot(config, tmp_path, monkeypatch):
    source = tmp_path / "narration.txt"
    source.write_bytes(("\ufeff \r\n" + narration() + "\r\n ").encode("utf-8"))
    pipeline = Pipeline.create(config, script_path=source)
    provider = FakeProviders()
    pipeline.providers = provider
    monkeypatch.setattr(media, "render", lambda store: None)
    monkeypatch.setattr(media, "finish", lambda store: None)
    monkeypatch.setattr(media, "validate", lambda store: {"passed": True})
    pipeline.run()
    assert provider.spoken == narration()
    assert pipeline.store.manifest.mode == "script"
    assert not {"research", "script", "revision", "audit"} & set(provider.calls)
    assert pipeline.store.path("input_script.txt").read_text(encoding="utf-8") == narration()
    assert pipeline.store.path("script.txt").read_text(encoding="utf-8") == narration()
    transcript = read_json(pipeline.store.path("transcript.json"))
    alignment = read_json(pipeline.store.path("alignment.json"))
    assert transcript and all(c["start"] < c["end"] for c in transcript)
    assert " ".join(c["text"] for c in transcript) == " ".join(narration().split())
    assert "".join(alignment["characters"]["characters"]) == narration()
    assert len(alignment["words"]) == len(narration().split())
    before = list(provider.calls)
    source.unlink()
    # Rebuild derived script files from the saved input without buying speech again.
    pipeline.store.path("script.txt").unlink()
    pipeline.store.path("script.json").unlink()
    resumed = Pipeline(Store(pipeline.store.folder, config), provider)
    resumed.run()
    assert provider.calls == before
    assert resumed.store.manifest.status == "completed"
    assert resumed.store.path("script.txt").read_text(encoding="utf-8") == narration()


def test_script_identity_reuses_normalized_content_and_rejects_job_collision(config, tmp_path):
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    original = Pipeline.create(config, script_path=source)
    copy = tmp_path / "another_name.txt"
    copy.write_bytes(("\ufeff\r\n" + narration() + "\r\n").encode("utf-8"))
    assert Pipeline.create(config, script_path=copy).store.folder == original.store.folder
    before = original.store.path("input_script.txt").read_bytes()
    source.write_text(narration().replace("fire", "lake"), encoding="utf-8")
    assert Pipeline.create(config, script_path=source).store.folder != original.store.folder
    with pytest.raises(PipelineError, match="different topic, script, or mode"):
        Pipeline.create(config, job_id=original.store.manifest.job_id, script_path=source)
    with pytest.raises(PipelineError, match="different topic, script, or mode"):
        Pipeline.create(config, "A topic", original.store.manifest.job_id)
    assert original.store.path("input_script.txt").read_bytes() == before


def test_corrupt_script_snapshot_blocks_generation_and_can_be_restored(config, tmp_path):
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    store = Pipeline.create(config, script_path=source).store
    atomic_bytes(store.path("input_script.txt"), narration().replace("fire", "lake").encode("utf-8"))
    provider = FakeProviders()
    with pytest.raises(PipelineError, match="Saved input script hash mismatch"):
        Pipeline(Store(store.folder, config), provider).run()
    assert provider.calls == []
    restored = Pipeline.create(config, script_path=source)
    assert restored.script().narration == narration()


@pytest.mark.parametrize("length", [0, 4499, 5001])
def test_invalid_input_script_stops_before_job_creation(config, tmp_path, length):
    source = tmp_path / "invalid.txt"
    source.write_text("x" * length, encoding="utf-8")
    with pytest.raises(PipelineError, match="required 4500"):
        Pipeline.create(config, script_path=source)
    assert not config.jobs_dir.exists()


@pytest.mark.parametrize("length", [4500, 5000])
def test_input_script_accepts_exact_length_boundaries(config, tmp_path, length):
    source = tmp_path / "boundary.txt"
    source.write_text("é" * (length - 1) + ".", encoding="utf-8")
    pipeline = Pipeline.create(config, script_path=source)
    assert pipeline.script().character_count == length


@pytest.mark.parametrize("use_flag", [False, True])
def test_cli_defaults_to_script_mode_without_changing_old_config(config, tmp_path, monkeypatch, use_flag):
    from video_automation.cli import main
    config.voice_id = "fixture-voice"
    for name in config.generation_keys:
        monkeypatch.setenv(name, "fixture-key")
    settings = tmp_path / "config.json"
    data = config.model_dump(mode="json")
    data.pop("input_mode")
    settings.write_text(json.dumps(data), encoding="utf-8")
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    calls = []
    monkeypatch.setattr(Pipeline, "run", lambda pipeline: calls.append((pipeline.store.manifest.mode, pipeline.script())))
    args = ["--config", str(settings), "run"] + (["--script"] if use_flag else []) + [str(source)]
    assert main(args) == 0
    assert len(calls) == 1 and calls[0][0] == "script"
    assert calls[0][1].narration == narration()


def test_cli_config_topic_default_and_explicit_script_override(config, tmp_path, monkeypatch):
    from video_automation.cli import main
    config.input_mode = "topic"
    config.voice_id = "fixture-voice"
    for name in config.generation_keys:
        monkeypatch.setenv(name, "fixture-key")
    settings = tmp_path / "config.json"
    settings.write_text(config.model_dump_json(), encoding="utf-8")
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    calls = []
    monkeypatch.setattr(Pipeline, "run", lambda pipeline: calls.append(pipeline.store.manifest.mode))
    assert main(["--config", str(settings), "run", "A topic"]) == 0
    assert main(["--config", str(settings), "run", "--script", str(source)]) == 0
    assert main(["--config", str(settings), "run", "--mode", "script", str(source)]) == 0
    assert calls == ["production", "script", "script"]


@pytest.mark.parametrize("inputs", [[], ["ignored", "--script", "file.txt"], ["--mode", "topic", "--script", "file.txt"]])
def test_cli_rejects_missing_or_ambiguous_input(config, tmp_path, monkeypatch, capsys, inputs):
    from video_automation.cli import main
    settings = tmp_path / "config.json"
    settings.write_text(config.model_dump_json(), encoding="utf-8")
    calls = []
    monkeypatch.setattr(Pipeline, "run", lambda pipeline: calls.append(1))
    assert main(["--config", str(settings), "run", *inputs]) == 1
    assert calls == []
    assert "Error:" in capsys.readouterr().err
    assert not list(config.jobs_dir.glob("*/manifest.json"))


def test_existing_manifest_resumes_topic_workflow_despite_script_default(store, monkeypatch):
    manifest = store.manifest.model_dump(mode="json")
    manifest.pop("script_input_hash")
    store.path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    restored = Store(store.folder, store.config)
    assert restored.config.input_mode == "script"
    assert restored.manifest.mode == "production"
    provider = FakeProviders("Too short.")
    with pytest.raises(PipelineError, match="required 4500"):
        Pipeline(restored, provider).run()
    assert provider.calls == ["research", "script", "revision"]


def visual_words(store, count=200):
    store.manifest.audio_duration = count
    return [Word(text=f"word{i}.", start=i, end=i+0.5, char_start=i*10, char_end=i*10+8,
                 sentence_id=f"sentence_{i}") for i in range(count)]


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unexpected", "reordered", "blank", "character", "schema"])
def test_visual_batch_retries_invalid_plan_before_checkpoint(store, invalid):
    provider = FakeProviders()
    original = provider.visuals
    def visuals(cues, previous_scenes=None):
        result = original(cues, previous_scenes)
        if provider.calls.count("visuals") == 1:
            if invalid == "missing":
                result["cues"].pop()
            elif invalid == "duplicate":
                result["cues"][-1]["id"] = result["cues"][0]["id"]
            elif invalid == "unexpected":
                result["cues"][-1]["id"] = "..."
            elif invalid == "reordered":
                result["cues"].reverse()
            elif invalid == "blank":
                result["cues"][0]["scene_description"] = " "
            elif invalid == "character":
                result["cues"][0]["character_ids"] = ["unknown"]
            else:
                del result["cues"][0]["motion"]
        return result
    provider.visuals = visuals
    words = visual_words(store, 8)
    Pipeline(store, provider).visuals(words)
    stage = store.manifest.stages["visuals.000"]
    assert stage.status == "completed" and stage.attempts == 2
    assert provider.calls == ["visuals", "visuals"]
    assert [c["id"] for c in read_json(store.path("transcript.json"))] == ["cue_0001", "cue_0002"]


def test_invalid_visual_plan_exhausts_retries_without_completed_checkpoint(store):
    provider = FakeProviders()
    original = provider.visuals
    def visuals(cues, previous_scenes=None):
        result = original(cues, previous_scenes)
        result["cues"] = result["cues"][:1]
        result["cues"][0]["id"] = "..."
        return result
    provider.visuals = visuals
    words = visual_words(store, 8)
    with pytest.raises(TemporaryError, match="every cue ID"):
        Pipeline(store, provider).visuals(words)
    stage = store.manifest.stages["visuals.000"]
    assert stage.status == "failed" and stage.attempts == store.config.max_attempts
    assert not store.path("visuals/batch_000.json").exists()
    assert not store.path("transcript.json").exists()
    with pytest.raises(PipelineError, match="retry limit reached"):
        Pipeline(Store(store.folder, store.config), provider).visuals(words)
    assert provider.calls == ["visuals"] * store.config.max_attempts


def test_resume_rejects_old_invalid_visual_cache_and_keeps_valid_work(store):
    from video_automation.providers import STYLE, VISUAL_PROMPT
    words = visual_words(store)
    cues = segment_words(words, store.config.fps, store.manifest.audio_duration)
    seed = FakeProviders()
    pipeline = Pipeline(store, seed)
    records = []
    for offset in range(0, len(cues), 24):
        batch = cues[offset:offset+24]
        previous_scenes = records[-3:]
        planned = seed.visuals(batch, previous_scenes)
        if offset == 24:
            planned["cues"] = planned["cues"][:2]
            for record in planned["cues"]:
                record["id"] = "..."
        pipeline.json_task(f"visuals.{offset//24:03}", {"cues": [c.model_dump() for c in batch],
                           "previous_scenes": previous_scenes, "text_generation": store.config.text_cache_settings,
                           "style": STYLE, "prompt": VISUAL_PROMPT}, f"visuals/batch_{offset//24:03}.json", lambda: planned)
        records.extend(planned["cues"])
    atomic_bytes(store.path("audio/narration.wav"), b"completed narration")
    atomic_bytes(store.path("alignment.json"), b"completed alignment")
    atomic_bytes(store.path("responses/visuals_bad.json"), b"preserved provider response")
    preserved = {path: file_hash(store.path(path)) for path in (
        "audio/narration.wav", "alignment.json", "visuals/batch_000.json", "responses/visuals_bad.json")}
    provider = FakeProviders()
    requested = []
    original = provider.visuals
    def visuals(batch, previous_scenes=None):
        requested.append(batch[0].id)
        return original(batch, previous_scenes)
    provider.visuals = visuals
    resumed = Pipeline(Store(store.folder, store.config), provider)
    resumed.visuals(words)
    assert requested == ["cue_0025", "cue_0049"]
    assert resumed.store.manifest.stages["visuals.000"].attempts == 1
    assert resumed.store.manifest.stages["visuals.001"].attempts == 2
    assert all(file_hash(store.path(path)) == expected for path, expected in preserved.items())
    assert [c.id for c in resumed.store.manifest.cues] == [c.id for c in cues]
    resumed.visuals(words)
    assert requested == ["cue_0025", "cue_0049"]
