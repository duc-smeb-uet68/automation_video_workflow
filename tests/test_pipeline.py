import base64
import io
import json
import math
import struct
import wave
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageChops, ImageDraw
from pydantic import ValidationError

from video_automation import assets, media
from video_automation.models import Config, Cue, Manifest, Word
from video_automation.pipeline import Pipeline
from video_automation.providers import Providers
from video_automation.store import PipelineError, Store, TemporaryError, atomic_bytes, digest, file_hash, read_json, worker_lock, write_json
from video_automation.timing import aligned_words, assign_frames, make_script, normalize, segment_words


@pytest.fixture
def config(tmp_path):
    return Config.load().model_copy(update={"image_mode": "api", "jobs_dir": tmp_path / "jobs", "width": 640, "height": 360,
                                           "image_interval_seconds": 0, "retry_backoff_seconds": 0, "max_attempts": 2})


@pytest.fixture
def store(config):
    return Store(config.jobs_dir / "test", config, Manifest(job_id="test", topic="Ancient caregivers", image_strategy="native_text"))


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

    def script(self, topic, research):
        self.calls.append("script")
        return {"narration": self.text, "source_references": ["https://example.org/study"]}

    def audit(self, script, research):
        self.calls.append("audit")
        return {"sentences": [{"sentence_id": s.id, "supported": True, "kind": "reconstruction",
                               "source_urls": [], "reason": "Explicit reconstruction fixture"} for s in script.sentences]}

    def metadata(self, narration):
        self.calls.append("metadata")
        return {"title": "Caregivers Beside the Fire", "description": "Imagine a parent and another caregiver watching a child."}

    def thumbnail(self, narration, title):
        self.calls.append("thumbnail")
        return {"headline": "Who watches the child?", "prompt": "An illustrative caregiver holding a child beside a fire."}

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
                          "scene_description": "Parent beside a fire.", "motion": "hold"} for c in cues]}

    def image(self, prompt):
        self.calls.append("image")
        return image_bytes()


def test_normalized_unicode_count_without_length_bounds():
    assert normalize(" \r\nA\rB\r\n ") == "A\nB"
    for length in (1, 4499, 4500, 5000, 5001, 50001):
        text = "é" * (length-1) + "."
        assert make_script(text).character_count == length
    script = make_script("You hear a cry.\n\nYou imagine a camp!")
    assert [s.id for s in script.sentences] == ["sentence_0001", "sentence_0002"]
    assert [s.paragraph_id for s in script.sentences] == ["paragraph_001", "paragraph_002"]
    assert all(script.narration[s.char_start:s.char_end] == s.text for s in script.sentences)


@pytest.mark.parametrize("text", ["", " \r\n\t "])
def test_empty_topic_narration_blocks_paid_downstream(store, text):
    provider = FakeProviders()
    provider.text = text
    pipeline = Pipeline(store, provider)
    with pytest.raises(PipelineError, match="Narration cannot be empty"):
        pipeline.run()
    assert provider.calls == ["research", "script"]
    assert store.manifest.status == "failed"
    assert store.path("failure_report.json").is_file()
    # Restart does not repeat completed research or rewrite the empty draft.
    with pytest.raises(PipelineError, match="Narration cannot be empty"):
        Pipeline(Store(store.folder, store.config), provider).run()
    assert provider.calls == ["research", "script"]


@pytest.mark.parametrize("text", ["You hear a cry.", "You imagine " + "é" * 6000 + "."], ids=["short", "long"])
def test_topic_narration_has_no_length_revision_and_reuses_cache(store, text):
    provider = FakeProviders(text)
    pipeline = Pipeline(store, provider)
    script = pipeline.script()
    assert script.narration == text and script.character_count == len(text)
    assert provider.calls == ["research", "script", "audit"]
    assert Pipeline(Store(store.folder, store.config), provider).script() == script
    assert provider.calls == ["research", "script", "audit"]
    assert "script.revision" not in store.manifest.stages


def test_alignment_requires_same_text_and_valid_timestamps():
    script = make_script("You hear.")
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


@pytest.mark.parametrize("text", ["", " You hear. ", "You\r\nhear."])
def test_speech_still_blocks_empty_or_unnormalized_narration(store, text):
    script = make_script("You hear.").model_copy(update={"narration": text})
    provider = FakeProviders()
    with pytest.raises(PipelineError, match="narration must be nonempty and normalized"):
        Pipeline(store, provider).speech(script)
    assert provider.calls == []


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
    def image(prompt):
        calls.append(1)
        return b"invalid-image" if len(calls) == 1 else original(prompt)
    provider.image = image
    store.manifest.cues = [Cue(id="cue_0001", text="Scene.", start=0, end=1, word_start=0, word_end=1,
                              scene_group="camp", scene_description="Parent by fire.")]
    Pipeline(store, provider).images()
    assert "image.character_reference" not in store.manifest.stages
    assert store.manifest.stages["image.cue_0001"].attempts == 2
    assert media.check_image(store.path("images/cue_0001.png"))["width"] == 640


def test_interrupted_image_preserves_previous_images_and_speech(store):
    provider = FakeProviders()
    pipeline = Pipeline(store, provider)
    atomic_bytes(store.path("audio/narration.wav"), b"unchanged-voiceover")
    voice_hash = file_hash(store.path("audio/narration.wav"))
    store.manifest.cues = [Cue(id=f"cue_{i:04}", text="A scene.", start=i-1, end=i,
                             word_start=(i-1)*2, word_end=i*2, scene_group="camp", scene_description=f"scene {i}") for i in range(1, 4)]
    original = provider.image
    def interrupt(prompt):
        if "scene 2" in prompt:
            raise KeyboardInterrupt
        return original(prompt)
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


@pytest.mark.parametrize("invalid", ["blank", "duplicate", "empty"])
def test_invalid_prompt_inputs_block_all_image_requests(store, invalid):
    provider = FakeProviders()
    store.manifest.cues = [Cue(id=f"cue_{index:04}", text="A scene.", start=index-1, end=index,
        word_start=index-1, word_end=index, scene_description="Scientist with a blue coat at a lab bench.")
        for index in range(1, 3)]
    if invalid == "blank":
        store.manifest.cues[-1].scene_description = " "
    elif invalid == "duplicate":
        store.manifest.cues[-1].id = store.manifest.cues[0].id
    else:
        store.manifest.cues = []
    with pytest.raises(PipelineError, match="Image prompts require"):
        Pipeline(store, provider).images()
    assert provider.calls == []
    assert not store.path("image_prompts.json").exists()
    assert not store.path("images").exists()


def test_prompt_change_regenerates_only_affected_image_and_revalidates_cache(store):
    provider = FakeProviders()
    store.manifest.cues = [Cue(id=f"cue_{index:04}", text="A scene.", start=index-1, end=index,
        word_start=index-1, word_end=index, scene_group="lab",
        scene_description=f"Scientist with a blue coat examines sample {index}.") for index in range(1, 3)]
    pipeline = Pipeline(store, provider)
    pipeline.images()
    assert provider.calls == ["image", "image"]
    stage = store.manifest.stages["image.prompts"]
    stage.result["data"].reverse()
    pipeline.images()
    assert stage.attempts == 2
    assert provider.calls == ["image", "image"]
    assert read_json(store.path("image_prompts.json"))[0]["cue_id"] == "cue_0001"
    store.manifest.cues[0].scene_description = "Scientist with a blue coat holds a red sample."
    pipeline.images()
    assert provider.calls == ["image", "image", "image"]
    assert store.manifest.stages["image.cue_0002"].attempts == 1
    assert read_json(store.path("image_prompts.json"))[0]["prompt"].startswith("Scientist with a blue coat holds a red sample.")


@pytest.mark.parametrize("mode", ["script", "production"])
@pytest.mark.parametrize("prefix", ["video_", "video_text_only_"])
def test_native_text_default_jobs_are_distinct_from_legacy_jobs(config, tmp_path, mode, prefix):
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    identity = {"mode": "script", "narration": narration()} if mode == "script" else "Caregivers"
    old_folder = config.jobs_dir / (prefix + digest(identity)[:12])
    legacy = Manifest(job_id=old_folder.name, topic="Caregivers", mode=mode).model_dump(mode="json")
    legacy.pop("image_strategy")
    legacy["schema_version"] = 1
    write_json(old_folder / "manifest.json", legacy)
    before = (old_folder / "manifest.json").read_bytes()
    pipeline = Pipeline.create(config, script_path=source) if mode == "script" else Pipeline.create(config, "Caregivers")
    assert pipeline.store.manifest.job_id == "video_native_text_" + digest(identity)[:12]
    saved = read_json(pipeline.store.path("manifest.json"))
    assert saved["image_strategy"] == "native_text" and saved["schema_version"] == 2
    assert (old_folder / "manifest.json").read_bytes() == before
    assert Store(pipeline.store.folder, config).manifest.image_strategy == "native_text"


@pytest.mark.parametrize("mode", ["script", "production", "fixture", "imported"])
@pytest.mark.parametrize("strategy", [None, "reference", "text_only"])
@pytest.mark.parametrize("command", ["run", "resume", "validate", "export-kdenlive"])
def test_legacy_jobs_are_rejected_without_writes(config, tmp_path, monkeypatch, capsys, mode, strategy, command):
    from video_automation.cli import main
    source = tmp_path / "narration.txt"
    source.write_text(narration(), encoding="utf-8")
    legacy_folder = config.jobs_dir / "legacy"
    manifest = Manifest(job_id="legacy", topic="Caregivers", mode=mode,
                        script_input_hash=digest(narration()) if mode == "script" else None).model_dump(mode="json")
    manifest["schema_version"] = 1
    manifest["cues"] = [{"id": "cue_0001", "overlay_text": "38%", "overlay_path": "overlays/cue_0001.png"}]
    if strategy is None:
        manifest.pop("image_strategy")
        manifest.pop("schema_version")
    else:
        manifest["image_strategy"] = strategy
    write_json(legacy_folder / "manifest.json", manifest)
    atomic_bytes(legacy_folder / "input_script.txt", narration().encode("utf-8"))
    atomic_bytes(legacy_folder / "images/cue_0001.png", image_bytes())
    atomic_bytes(legacy_folder / "audio/narration.wav", b"saved audio")
    atomic_bytes(legacy_folder / "overlays/cue_0001.png", b"saved overlay")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in legacy_folder.rglob("*") if path.is_file()}
    settings = tmp_path / "config.json"
    settings.write_text(config.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(Pipeline, "run", lambda pipeline: pytest.fail("Legacy job must not run"))
    if command == "run":
        args = (["--script", str(source)] if mode == "script" else ["--mode", "topic", "Caregivers"])
        args += ["--job-id", "legacy"]
    else:
        args = ["legacy"]
        if command == "resume":
            args += ["--retry-failed"]
        elif command == "export-kdenlive":
            args += ["--template", str(tmp_path / "template.kdenlive")]
    assert main(["--config", str(settings), command, *args]) == 1
    error = capsys.readouterr().err
    assert "Existing job is preserved" in error
    assert "codex/text-only-image-prompts" in error and "main" in error
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in legacy_folder.rglob("*") if path.is_file()}


@pytest.mark.parametrize("version", [None, 1, 3])
def test_unsupported_schema_rejected_before_parsing_or_creating_files(config, version):
    folder = config.jobs_dir / "old"
    saved = {"schema_version": version, "mode": "fixture", "cues": [{"overlay_text": "38%"}]}
    if version is None:
        saved.pop("schema_version")
    write_json(folder / "manifest.json", saved)
    before = (folder / "manifest.json").read_bytes(), (folder / "manifest.json").stat().st_mtime_ns
    with pytest.raises(PipelineError, match="Existing job is preserved"):
        Store(folder, config)
    assert before == ((folder / "manifest.json").read_bytes(), (folder / "manifest.json").stat().st_mtime_ns)
    assert list(folder.iterdir()) == [folder / "manifest.json"]


@pytest.mark.parametrize("strategy", [None, "reference", "text_only"])
def test_version_two_requires_native_text_strategy_for_generated_jobs(config, strategy):
    folder = config.jobs_dir / "wrong_strategy"
    saved = Manifest(job_id=folder.name, topic="A topic").model_dump(mode="json")
    if strategy is None:
        saved.pop("image_strategy")
    else:
        saved["image_strategy"] = strategy
    write_json(folder / "manifest.json", saved)
    before = (folder / "manifest.json").read_bytes()
    with pytest.raises(PipelineError, match="native-text generation"):
        Store(folder, config)
    assert (folder / "manifest.json").read_bytes() == before


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
    assert store.manifest.script_character_count == len(provider.spoken)
    before = list(provider.calls)
    Pipeline(Store(store.folder, store.config), provider).run()
    assert provider.calls == before
    assert store.manifest.status == "completed"
    assert len(store.manifest.cues) != 81


@pytest.mark.parametrize("mode", ["production", "script"])
@pytest.mark.parametrize("text", ["You hear a cry.\n\nYou imagine a camp!", "You imagine " + "é" * 6000 + "."],
                         ids=["short", "long"])
def test_offline_full_pipeline_renders_real_mp4(store, tmp_path, mode, text, monkeypatch):
    import video_automation.progress as tracking
    snapshots = []
    original_write = tracking.write_json

    def capture(path, content):
        snapshots.append(json.loads(json.dumps(content)))
        original_write(path, content)

    monkeypatch.setattr(tracking, "write_json", capture)
    provider = FakeProviders(text)
    if mode == "script":
        path = tmp_path / "narration.txt"
        path.write_text(text, encoding="utf-8")
        store = Pipeline.create(store.config, script_path=path).store
    pipeline = Pipeline(store, provider)
    report = pipeline.run()
    assert report["passed"] and not report["details"]["narration_length"]["enforced"]
    assert report["details"]["narration_length"]["integrity_checked"]
    assert store.manifest.script_character_count == len(text)
    assert provider.spoken == text
    assert len(store.manifest.cues) == len(list(store.path("images").glob("cue_*.png")))
    assert store.path("final.mp4").is_file()
    assert store.path("subtitles.en.srt").is_file()
    assert store.manifest.total_frames == 240
    assert report["mode"] == mode
    progress = read_json(store.path("progress.json"))
    assert progress["status"] == "completed"
    phases = ["narration", "metadata", "speech", "visuals", "images", "render", "finish", "validate"]
    if store.manifest.thumbnail_enabled:
        phases.insert(2, "thumbnail")
    assert progress["completed_phases"] == progress["total_phases"] == len(phases)
    assert list(dict.fromkeys(snapshot["phase"] for snapshot in snapshots)) == phases
    assert read_json(store.path("video_metadata.json"))["title"] == "Caregivers Beside the Fire"
    for phase, total in (("visuals", (len(store.manifest.cues)+23)//24),
                         ("images", len(store.manifest.cues)), ("render", len(store.manifest.cues))):
        completed = [snapshot for snapshot in snapshots if snapshot["phase"] == phase and snapshot["phase_status"] == "completed"][-1]
        assert completed["completed_units"] == completed["total_units"] == total
    if mode == "script":
        assert report["details"]["source_audit"] == "Skipped: narration supplied by the user."
        assert not {"research", "script", "revision", "audit"} & set(provider.calls)
        assert not store.path("sources.json").exists()
        assert not store.path("source_audit.json").exists()
        # Script mode still requires fresh speech alignment and unchanged narration.
        store.path("alignment.json").unlink()
        with pytest.raises(PipelineError, match="alignment_audio_match"):
            media.validate(store)
        atomic_bytes(store.path("script.txt"), b"Changed narration.")
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
    assert "overlay_bounds" not in report["checks"]
    assert not pipeline.store.path("overlays").exists()
    assert not any(name.startswith("overlay.") for name in pipeline.store.manifest.stages)
    saved = read_json(pipeline.store.path("manifest.json"))
    assert saved["schema_version"] == 2
    assert all("overlay_text" not in cue and "overlay_path" not in cue for cue in saved["cues"])
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
    cue = pipeline.store.manifest.cues[-1]
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
    with Image.open(pipeline.store.path(f"render/prepared/{cue.id}.png")) as prepared, Image.open(pipeline.store.path(cue.image_path)) as original:
        assert ImageChops.difference(prepared, original).getbbox() is None
    assert not pipeline.store.path("overlays").exists()
    assert sum(len(image.findall("filter")) for image in images) == 1
    assert pipeline.store.manifest.schema_version == 2


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


@pytest.mark.parametrize("background", ["#fffaf0", "#143860", "#28b369"])
@pytest.mark.parametrize("size", [(640, 360), (1920, 1080)])
def test_prepare_image_preserves_native_text_and_all_source_pixels(config, tmp_path, background, size):
    config.width, config.height = size
    source, prepared = tmp_path / "source.png", tmp_path / "prepared.png"
    image = Image.new("RGB", size, background)
    ImageDraw.Draw(image).text((30, 30), "38%", fill=(255, 0, 0), font_size=32)
    image.save(source)
    media.prepare_image(config, source, prepared)
    with Image.open(prepared) as result:
        assert ImageChops.difference(image, result).getbbox() is None


def test_native_text_fixture_survives_video_render_and_cached_resume(config, tmp_path):
    pipeline = assets.fixture(config, "native_text_fixture")
    store = pipeline.store
    cue = store.manifest.cues[-1]
    with Image.open(store.path(cue.image_path)) as source:
        expected_red = {(x, y) for x in range(30, 120) for y in range(20, 70)
                        if source.getpixel((x, y)) == (160, 25, 20)}
    assert len(expected_red) > 100
    assert pipeline.run()["passed"]
    frame = tmp_path / "native_text_frame.png"
    media.ffmpeg(config, "-loglevel", "error", "-ss", "2.5", "-i", store.path("final.mp4"),
                 "-frames:v", 1, "-threads", 1, frame)
    with Image.open(frame) as decoded:
        preserved_red = sum(1 for point in expected_red
                            if decoded.getpixel(point)[0] > decoded.getpixel(point)[1] + 50
                            and decoded.getpixel(point)[0] > decoded.getpixel(point)[2] + 50)
    assert preserved_red >= len(expected_red) * 0.95
    before = {name: file_hash(store.path(name)) for name in (
        "audio/narration.wav", cue.image_path, "render/prepared/cue_0003.png", "final.mp4", "subtitles.en.srt")}
    assert Pipeline(Store(store.folder, config)).run()["passed"]
    assert before == {name: file_hash(store.path(name)) for name in before}
    assert not store.path("overlays").exists()
    assert not any(name.startswith("overlay.") for name in store.manifest.stages)


@pytest.mark.parametrize("field", ["overlay_text", "overlay_path"])
def test_new_cue_schema_rejects_removed_overlay_fields(field):
    record = Cue(id="cue_0001", text="A scene.", start=0, end=1, word_start=0, word_end=2).model_dump()
    record[field] = "old overlay"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Cue.model_validate(record)


def test_native_text_visual_plan_and_images_preserve_planned_wording(store):
    provider = FakeProviders()
    original = provider.visuals
    def visuals(cues, previous_scenes=None):
        planned = original(cues, previous_scenes)
        planned["cues"][0]["scene_description"] += ' Draw the exact label "38%" within the image in a suitable location.'
        planned["cues"][1]["motion"] = "slow_push"
        return planned
    provider.visuals = visuals
    pipeline = Pipeline(store, provider)
    words = visual_words(store, 8)
    pipeline.visuals(words)
    assert len(store.manifest.cues) == 2
    assert [cue.motion for cue in store.manifest.cues] == ["hold", "slow_push"]
    pipeline.images()
    prompts = read_json(store.path("image_prompts.json"))
    assert '"38%"' in prompts[0]["prompt"] and '"38%"' not in prompts[1]["prompt"]
    assert all("overlay_text" not in cue and "overlay_path" not in cue for cue in read_json(store.path("transcript.json")))
    before = list(provider.calls)
    resumed = Pipeline(Store(store.folder, store.config), provider)
    resumed.visuals(words)
    resumed.images()
    assert provider.calls == before
    assert not store.path("overlays").exists()


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


@pytest.mark.parametrize("text", ["", " \r\n\t "])
def test_empty_input_script_stops_before_job_creation(config, tmp_path, text):
    source = tmp_path / "invalid.txt"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(PipelineError, match="Narration cannot be empty"):
        Pipeline.create(config, script_path=source)
    assert not config.jobs_dir.exists()


@pytest.mark.parametrize("length", [1, 4499, 4500, 5000, 5001, 10001, 50001])
def test_input_script_accepts_nonempty_text_without_length_bounds(config, tmp_path, length):
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
    provider = FakeProviders("You hear a cry.")
    script = Pipeline(restored, provider).script()
    assert script.narration == "You hear a cry."
    assert provider.calls == ["research", "script", "audit"]


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
                result["cues"][0]["character_ids"] = [" "]
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


@pytest.mark.parametrize("exhaust", [False, True])
def test_incomplete_visual_output_retries_without_repeating_completed_work(store, monkeypatch, exhaust):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    words = visual_words(store)
    atomic_bytes(store.path("audio/narration.wav"), b"completed narration")
    atomic_bytes(store.path("alignment.json"), b"completed alignment")
    preserved = {path: file_hash(store.path(path)) for path in ("audio/narration.wav", "alignment.json")}
    calls, recovered = [], [False]
    def handler(request):
        payload = json.loads(request.content)
        ids = payload["text"]["format"]["schema"]["properties"]["cues"]["items"]["properties"]["id"]["enum"]
        calls.append(ids[0])
        if ids[0] == "cue_0025" and not recovered[0]:
            if not exhaust:
                recovered[0] = True
            return httpx.Response(200, json={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"cues": [\n' + ' '*100}]}]})
        plan = {"cues": [{"id": value, "scene_group": "camp", "character_ids": [],
                         "scene_description": "A campfire.", "motion": "hold"} for value in ids]}
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message",
            "content": [{"type": "output_text", "text": json.dumps(plan)}]}]})
    original_client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs))
    if exhaust:
        with pytest.raises(TemporaryError, match="max_output_tokens"):
            Pipeline(store).visuals(words)
        assert not store.path("visuals/batch_001.json").exists()
        assert calls == ["cue_0001", "cue_0025", "cue_0025"]
        resumed = Store(store.folder, store.config)
        with pytest.raises(PipelineError, match="retry limit reached"):
            Pipeline(resumed).visuals(words)
        assert len(calls) == 3
        recovered[0] = True
        resumed.manifest.stages["visuals.001"].attempts = 0
        resumed.save()
        Pipeline(resumed).visuals(words)
        assert calls == ["cue_0001", "cue_0025", "cue_0025", "cue_0025", "cue_0049"]
    else:
        Pipeline(store).visuals(words)
        resumed = Store(store.folder, store.config)
        assert calls == ["cue_0001", "cue_0025", "cue_0025", "cue_0049"]
        assert resumed.manifest.stages["visuals.001"].attempts == 2
    assert resumed.manifest.stages["visuals.000"].attempts == 1
    assert all(file_hash(store.path(path)) == expected for path, expected in preserved.items())
    assert len(resumed.manifest.usage) == len(calls)
    Pipeline(resumed).visuals(words)
    assert len(resumed.manifest.usage) == len(calls)


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
