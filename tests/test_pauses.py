import base64
import io
import wave

import pytest
from pydantic import ValidationError

from test_pipeline import FakeProviders, image_bytes
from video_automation import media
from video_automation.models import Config, Manifest, ScriptPause
from video_automation.pipeline import Pipeline
from video_automation.store import PipelineError, Store, atomic_bytes, digest, file_hash, read_json, write_json
from video_automation.timing import SPEECH_SAMPLE_RATE, aligned_words, make_script, normalize, parse_script_input, pause_timing


@pytest.fixture
def config(tmp_path):
    return Config.load().model_copy(update={"jobs_dir": tmp_path / "jobs", "width": 640, "height": 360,
        "image_interval_seconds": 0, "retry_backoff_seconds": 0, "max_attempts": 2})


def create_pipeline(config, tmp_path, text, *, thumbnail=False):
    source = tmp_path / "narration.txt"
    source.write_bytes(("\ufeff\r\n" + text + "\r\n").encode("utf-8"))
    pipeline = Pipeline.create(config, script_path=source, thumbnail_enabled=thumbnail)
    pipeline.providers = FakeProviders()
    return pipeline, source


def read_wav(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes(), audio.readframes(audio.getnframes())


def test_parser_preserves_normalized_text_and_records_word_boundaries():
    text = "[pause=0.25s]\r\nThink about that. [pause=1s] [pause=0.5s]\r\nNow continue.\r\n[pause=0.75s]"
    narration, pauses = parse_script_input(text)
    assert narration == "Think about that.  \nNow continue."
    assert [(p.word_index, p.seconds) for p in pauses] == [(0, 0.25), (3, 1), (3, 0.5), (5, 0.75)]
    plain = " \r\nUnicode é stays.\r\n\r\nAnd  double spaces. \r\n"
    assert parse_script_input(plain) == (normalize(plain), [])


@pytest.mark.parametrize("marker", ["[pause]", "[pause=]", "[pause=0s]", "[pause=-1s]", "[pause=61s]",
    "[pause=0.000001s]", "[pause=nans]", "[pause=infs]", "[pause=1]", "[pause=1ms]", "[PAUSE=1s]",
    "[pause=1s", "[pause=1s\n", "[pause=1\ns]", "[pause=1.0.0s]", "[pause =1s]"])
def test_invalid_markers_rejected_before_job_creation(config, tmp_path, marker):
    source = tmp_path / "invalid.txt"
    source.write_text("Think. " + marker + " Continue.", encoding="utf-8")
    with pytest.raises(PipelineError, match="Invalid pause marker"):
        Pipeline.create(config, script_path=source)
    assert not config.jobs_dir.exists()


@pytest.mark.parametrize("text", ["hel[pause=1s]lo", "Think.[pause=1s] Continue.",
    "Think. [pause=1s]Continue.", "Think. [pause=1s][pause=1s] Continue."])
def test_pause_markers_require_whitespace_boundaries(text):
    with pytest.raises(PipelineError, match="separated from words"):
        parse_script_input(text)


def test_pause_only_input_is_not_narration(config, tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("[pause=1s]\n[pause=2s]", encoding="utf-8")
    with pytest.raises(PipelineError, match="Narration cannot be empty"):
        Pipeline.create(config, script_path=source)
    assert not config.jobs_dir.exists()


@pytest.mark.parametrize("updates", [{"mode": "production", "script_pauses": [{"word_index": 0, "seconds": 1}]},
    {"mode": "script", "script_pauses": [{"word_index": 2, "seconds": 1}, {"word_index": 1, "seconds": 1}]},
    {"mode": "script", "script_pauses": [{"word_index": 0, "seconds": float("nan")}]},
    {"mode": "script", "script_pauses": [{"word_index": 0, "seconds": 61}]}])
def test_manifest_rejects_invalid_pause_instructions(updates):
    with pytest.raises(ValidationError):
        Manifest(job_id="invalid", topic="Invalid", **updates)


def test_speech_inserts_exact_samples_and_shifts_verified_alignment(config, tmp_path):
    text = "[pause=0.25s]\nThink about that. [pause=1s] [pause=0.5s]\nNow continue.\n[pause=0.75s]"
    pipeline, _ = create_pipeline(config, tmp_path, text)
    store = pipeline.store
    script = pipeline.script()
    words = pipeline.speech(script)
    assert pipeline.providers.spoken == script.narration and "[pause" not in script.narration
    assert store.path("input_script.txt").read_text(encoding="utf-8") == text
    assert store.manifest.script_character_count == len(script.narration)
    response = read_json(store.path("audio/speech_response.json"))
    with wave.open(io.BytesIO(base64.b64decode(response["audio_base64"])), "rb") as original:
        original_frames = original.getnframes()
        original_samples = original.readframes(original_frames)
    original_words = aligned_words(script, response["alignment"], original_frames / SPEECH_SAMPLE_RATE)
    frames, samples = read_wav(store.path("audio/narration.wav"))
    alignment = read_json(store.path("alignment.json"))
    pauses = alignment["pauses"]
    assert frames == original_frames + round(2.5 * SPEECH_SAMPLE_RATE)
    assert store.manifest.audio_duration == frames / SPEECH_SAMPLE_RATE == 10.5
    restored, cursor = bytearray(), 0
    for pause in pauses:
        start = round(pause["start"] * SPEECH_SAMPLE_RATE) * 2
        end = start + pause["frames"] * 2
        assert samples[start:end] == b"\0" * (end - start)
        restored.extend(samples[cursor:start])
        cursor = end
    restored.extend(samples[cursor:])
    assert restored == original_samples
    for index, (before, after) in enumerate(zip(original_words, words)):
        shift = sum(p.seconds for p in store.manifest.script_pauses if p.word_index <= index)
        assert after.text == before.text
        assert after.start == pytest.approx(before.start + shift)
        assert after.end == pytest.approx(before.end + shift)
    assert media.validate_pauses(store, script.narration, alignment) == pauses
    pipeline.visuals(words)
    assert len(store.manifest.cues) == 2
    first, second = store.manifest.cues
    assert first.word_end == second.word_start == 3
    assert first.end_frame == second.start_frame == round(second.start * config.fps)
    assert first.end <= pauses[1]["start"] + 1 / SPEECH_SAMPLE_RATE
    assert second.start >= pauses[2]["end"]
    assert "[pause" not in store.path("subtitles.en.srt").read_text(encoding="utf-8")


@pytest.mark.parametrize("music", [False, True])
def test_external_pause_workflow_renders_valid_video_and_cached_resume(config, tmp_path, music):
    text = "Take a moment. [pause=4s] Then continue."
    pipeline, source = create_pipeline(config, tmp_path, text, thumbnail=True)
    store, provider = pipeline.store, pipeline.providers
    contexts = []
    metadata, thumbnail = provider.metadata, provider.thumbnail
    provider.metadata = lambda narration: (contexts.append(narration), metadata(narration))[1]
    provider.thumbnail = lambda narration, title: (contexts.append(narration), thumbnail(narration, title))[1]
    if music:
        config.music_path = tmp_path / "music.wav"
        media.ffmpeg(config, "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", config.music_path)
    assert pipeline.run()["status"] == "waiting_for_images"
    assert contexts == [provider.spoken, provider.spoken]
    assert all("[pause" not in context for context in contexts)
    assert "[pause" not in store.path("image_prompts.json").read_text(encoding="utf-8")
    assert len(store.manifest.cues) == 2
    assert "image" not in provider.calls
    for cue in store.manifest.cues:
        atomic_bytes(store.path(f"images/{cue.id}.png"), image_bytes())
    atomic_bytes(store.path("thumbnail.png"), image_bytes())
    source.unlink()
    calls = list(provider.calls)
    resumed = Pipeline(Store(store.folder, config), provider)
    report = resumed.run()
    assert report["passed"] and report["checks"]["intentional_pauses"]
    assert resumed.store.manifest.audio_duration == 12
    assert resumed.store.manifest.total_frames == 360
    assert report["checks"]["narration_no_long_silence"] and report["checks"]["no_long_silence"]
    assert report["details"]["narration_health"]["long_silences"]
    assert report["details"]["narration_health"]["unexpected_long_silences"] == []
    assert provider.calls == calls
    files = ["audio/narration.wav", "audio/speech_response.json", "alignment.json", "rough.mp4", "final.mp4"]
    before = {p: (file_hash(store.path(p)), store.path(p).stat().st_mtime_ns) for p in files}
    assert Pipeline(Store(store.folder, config), provider).run()["passed"]
    assert before == {p: (file_hash(store.path(p)), store.path(p).stat().st_mtime_ns) for p in files}
    assert provider.calls == calls
    assert all(stage.attempts == 1 for stage in resumed.store.manifest.stages.values())


def test_corrupt_paused_audio_rebuilds_from_cached_speech_without_duplicating_pause(config, tmp_path):
    pipeline, _ = create_pipeline(config, tmp_path, "Think. [pause=1s] Continue.")
    script = pipeline.script()
    pipeline.speech(script)
    store = pipeline.store
    expected = file_hash(store.path("audio/narration.wav"))
    atomic_bytes(store.path("audio/narration.wav"), b"corrupt")
    resumed = Pipeline(Store(store.folder, config), pipeline.providers)
    resumed.speech(resumed.script())
    assert file_hash(store.path("audio/narration.wav")) == expected
    assert resumed.store.manifest.audio_duration == 9
    assert pipeline.providers.calls == ["speech"]


def test_pause_duration_changes_job_identity_and_rejects_explicit_collision(config, tmp_path):
    pipeline, source = create_pipeline(config, tmp_path, "Think. [pause=1s] Continue.")
    before = {p: p.read_bytes() for p in pipeline.store.folder.rglob("*") if p.is_file()}
    source.write_text("Think. [pause=2s] Continue.", encoding="utf-8")
    assert Pipeline.create(config, script_path=source).store.folder != pipeline.store.folder
    with pytest.raises(PipelineError, match="different topic, script, or mode"):
        Pipeline.create(config, script_path=source, job_id=pipeline.store.manifest.job_id)
    assert before == {p: p.read_bytes() for p in pipeline.store.folder.rglob("*") if p.is_file()}


def test_old_jobs_keep_plain_speech_cache_and_do_not_acquire_pause_semantics(config, tmp_path):
    pipeline, _ = create_pipeline(config, tmp_path, "Think. Continue.")
    script = pipeline.script()
    pipeline.speech(script)
    store = pipeline.store
    saved = read_json(store.path("manifest.json"))
    saved.pop("script_pauses")
    write_json(store.path("manifest.json"), saved)
    before = {p: (file_hash(store.path(p)), store.path(p).stat().st_mtime_ns)
              for p in ("audio/narration.wav", "alignment.json")}
    resumed = Pipeline(Store(store.folder, config), pipeline.providers)
    resumed.speech(resumed.script())
    assert pipeline.providers.calls == ["speech"]
    assert before == {p: (file_hash(store.path(p)), store.path(p).stat().st_mtime_ns) for p in before}
    literal = "Think. [pause=1s] Continue."
    old = Store(config.jobs_dir / "old", config,
                Manifest(job_id="old", topic="Old", mode="script", script_input_hash=digest(literal)))
    atomic_bytes(old.path("input_script.txt"), literal.encode("utf-8"))
    assert Pipeline(old).script().narration == literal
    assert old.manifest.script_pauses == []


@pytest.mark.parametrize("corruption", ["timing", "samples", "snapshot", "instructions"])
def test_pause_validation_rejects_corrupt_data(config, tmp_path, corruption):
    pipeline, _ = create_pipeline(config, tmp_path, "Think. [pause=1s] Continue.")
    script = pipeline.script()
    pipeline.speech(script)
    store = pipeline.store
    alignment = read_json(store.path("alignment.json"))
    if corruption == "timing":
        alignment["pauses"][0]["end"] += 1
    elif corruption == "samples":
        path = store.path("audio/narration.wav")
        with wave.open(str(path), "rb") as audio:
            params = audio.getparams()
            content = bytearray(audio.readframes(audio.getnframes()))
        sample = round(alignment["pauses"][0]["start"] * SPEECH_SAMPLE_RATE)
        content[sample * 2:sample * 2 + 2] = b"\x01\x00"
        with wave.open(str(path), "wb") as audio:
            audio.setparams(params)
            audio.writeframes(content)
    elif corruption == "snapshot":
        atomic_bytes(store.path("input_script.txt"), b"Think. [pause=2s] Continue.")
    else:
        store.manifest.script_pauses[0].seconds = 2
    with pytest.raises(PipelineError, match="Pause timing|non-silent|Pause instructions"):
        media.validate_pauses(store, script.narration, alignment)


def test_intentional_silence_does_not_hide_other_long_silences():
    pauses = [{"start": 3, "end": 7}]
    assert media.unexpected_silences([{"start": 0, "end": 10}], pauses, 3) == [
        {"start": 0, "end": 3}, {"start": 7, "end": 10}]
    assert media.unexpected_silences([{"start": 2.99, "end": 7.01}], pauses, 3) == []
    intervals = [{"start": 10, "end": 14}]
    assert media.unexpected_silences(intervals, pauses, 3) == intervals
    assert media.unexpected_silences(intervals, [], 3) == intervals


def test_pause_does_not_bypass_exact_provider_text_checks():
    script = make_script("Think. Continue.")
    response = FakeProviders().speech(script.narration)
    response["alignment"]["characters"][0] = "X"
    with pytest.raises(PipelineError, match="exact normalized narration"):
        pause_timing(script, response["alignment"], 8 * SPEECH_SAMPLE_RATE, [ScriptPause(word_index=1, seconds=1)])


def test_failed_audio_insertion_preserves_original_wav(config, tmp_path):
    source = tmp_path / "original.wav"
    atomic_bytes(source, base64.b64decode(FakeProviders().speech("Think.")["audio_base64"]))
    before = file_hash(source)
    with pytest.raises(PipelineError, match="sample order"):
        media.insert_silence(source, [{"source_frame": 9 * SPEECH_SAMPLE_RATE, "frames": SPEECH_SAMPLE_RATE}])
    assert file_hash(source) == before
    assert not source.with_suffix(".pauses.tmp.wav").exists()
