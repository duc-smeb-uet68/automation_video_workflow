import base64
import json
import re
import sys
import types

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from test_pipeline import FakeProviders, image_bytes
from video_automation import media, speech
from video_automation.cli import doctor, main
from video_automation.models import Config
from video_automation.pipeline import Pipeline
from video_automation.providers import Providers
from video_automation.store import PipelineError, Store, TemporaryError, file_hash, read_json, write_json
from video_automation.timing import aligned_words, make_script, pause_timing


@pytest.fixture
def config(tmp_path):
    return Config.load().model_copy(update={"speech_provider": "voicestudio", "voicestudio_voice": "saved-voice",
        "jobs_dir": tmp_path / "jobs", "width": 640, "height": 360, "max_attempts": 2,
        "retry_backoff_seconds": 0, "image_mode": "external"})


def create(config, tmp_path, text="You see 38% growth. Keep going!"):
    source = tmp_path / "narration.txt"
    source.write_text(text, encoding="utf-8")
    pipeline = Pipeline.create(config, script_path=source, thumbnail_enabled=False)
    pipeline.providers = FakeProviders(text)
    return pipeline


def wav_bytes():
    return base64.b64decode(FakeProviders().speech("Test.")["audio_base64"])


def align_fixture(config, audio, script):
    duration = media.audio_duration(config, audio)
    matches = list(re.finditer(r"\S+", script.narration))
    step = (duration - 0.2) / len(matches)
    tokens = [{"word": m.group(), "start": i * step, "end": (i+1) * step} for i, m in enumerate(matches)]
    return speech.map_alignment(script, tokens, duration)


@pytest.fixture
def local_speech(monkeypatch):
    calls = []
    audio = wav_bytes()
    def synthesize(self, text):
        calls.append(text)
        return audio
    monkeypatch.setattr(speech.VoiceStudio, "synthesize", synthesize)
    monkeypatch.setattr(speech, "force_align", align_fixture)
    return calls


@pytest.mark.parametrize("text", ["A" * 4096, "B" * 9000, "A.  B\n" * 2000, "é 😀 38%!\n" * 1000])
def test_chunks_preserve_exact_unicode_and_request_limit(text):
    chunks = speech.speech_chunks(text)
    assert "".join(chunks) == text
    assert all(0 < len(chunk) <= 4096 for chunk in chunks)


def test_chunk_prefers_sentence_then_whitespace():
    assert speech.speech_chunks("Hello. Next words", 12) == ["Hello. ", "Next words"]
    assert speech.speech_chunks("hello world again", 12) == ["hello world ", "again"]


@pytest.mark.parametrize("url", ["https://example.org", "http://example.org", "http://user:secret@localhost:3900",
                                      "http://localhost:3900/path", "http://localhost:99999"])
def test_reject_nonlocal_or_credential_url(url):
    with pytest.raises(ValidationError):
        Config(voicestudio_base_url=url, speech_provider="voicestudio")


def mock_api(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(speech.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_local_http_contract_and_no_dollar_reservation(config, monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"paths": {route: {} for route in
                ("/v1/audio/speech", "/v1/audio/voices", "/v1/models")}})
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"voices": [{"voice_id": "saved-voice"}]})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "omnivoice"}]})
        assert json.loads(request.content) == {"input": "Hello, 38%!", "voice": "saved-voice", "model": "omnivoice",
                                               "response_format": "wav", "language": "en", "speed": 1.0}
        return httpx.Response(200, content=wav_bytes())
    mock_api(monkeypatch, handler)
    adapter = speech.VoiceStudio(config, config.speech_snapshot)
    assert adapter.synthesize("Hello, 38%!").startswith(b"RIFF")
    assert len(requests) == 4


@pytest.mark.parametrize("status,temporary", [(429, True), (503, True), (401, False), (422, False)])
def test_http_errors_redact_bearer(config, monkeypatch, status, temporary):
    monkeypatch.setenv("VOICESTUDIO_API_KEY", "test-secret-value")
    mock_api(monkeypatch, lambda request: httpx.Response(status, json={"detail": "failure test-secret-value"}))
    adapter = speech.VoiceStudio(config, config.speech_snapshot)
    adapter.ready = True
    with pytest.raises(TemporaryError if temporary else PipelineError) as error:
        adapter.synthesize("Hello.")
    assert "test-secret-value" not in str(error.value)
    assert f"HTTP {status}" in str(error.value)


@pytest.mark.parametrize("content", [b"", b"not audio", wav_bytes()[:80]])
def test_invalid_audio_is_retryable(config, monkeypatch, content):
    mock_api(monkeypatch, lambda request: httpx.Response(200, content=content))
    adapter = speech.VoiceStudio(config, config.speech_snapshot)
    adapter.ready = True
    with pytest.raises(TemporaryError, match="invalid or empty WAV"):
        adapter.synthesize("Hello.")


def test_lazy_launch_once_and_reuse(config, monkeypatch):
    adapter = speech.VoiceStudio(config, config.speech_snapshot)
    calls = []
    def inspect():
        calls.append("probe")
        if len(calls) == 1:
            raise speech.BackendUnavailable("offline")
    monkeypatch.setattr(adapter, "inspect", inspect)
    monkeypatch.setattr(adapter, "launch", lambda: calls.append("launch"))
    adapter.ensure_ready()
    adapter.ensure_ready()
    assert calls == ["probe", "launch", "probe"]


def test_startup_disabled_or_incompatible_does_not_launch(config, monkeypatch):
    adapter = speech.VoiceStudio(config, config.speech_snapshot.model_copy(update={"auto_launch": False}))
    monkeypatch.setattr(adapter, "inspect", lambda: (_ for _ in ()).throw(speech.BackendUnavailable("offline")))
    monkeypatch.setattr(adapter, "launch", lambda: pytest.fail("must not launch"))
    with pytest.raises(PipelineError, match="Open the installed app"):
        adapter.ensure_ready()
    adapter.settings.auto_launch = True
    monkeypatch.setattr(adapter, "inspect", lambda: (_ for _ in ()).throw(PipelineError("incompatible")))
    with pytest.raises(PipelineError, match="incompatible"):
        adapter.ensure_ready()


def test_startup_timeout(config, monkeypatch):
    adapter = speech.VoiceStudio(config, config.speech_snapshot)
    monkeypatch.setattr(adapter, "inspect", lambda: (_ for _ in ()).throw(speech.BackendUnavailable("offline")))
    monkeypatch.setattr(adapter, "launch", lambda: None)
    ticks = iter([0, 0, 181])
    monkeypatch.setattr(speech.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TemporaryError, match="180 seconds"):
        adapter.ensure_ready()


def test_launch_uses_executable_argument_and_hidden_window(config, tmp_path, monkeypatch):
    executable = tmp_path / "Voice Studio.exe"
    executable.write_bytes(b"test")
    settings = config.speech_snapshot.model_copy(update={"executable": executable})
    calls = []
    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    speech.VoiceStudio(config, settings).launch()
    args, options = calls[0]
    assert args == ([str(executable)],)
    assert "shell" not in options
    assert options["cwd"] == str(executable.parent)
    if speech.os.name == "nt":
        assert options["startupinfo"].wShowWindow == speech.subprocess.SW_HIDE


@pytest.mark.parametrize("missing", ["routes", "voice", "model"])
def test_inspect_rejects_incompatible_api_or_missing_setup(config, monkeypatch, missing):
    def handler(request):
        if request.url.path == "/openapi.json":
            paths = {} if missing == "routes" else {route: {} for route in
                ("/v1/audio/speech", "/v1/audio/voices", "/v1/models")}
            return httpx.Response(200, json={"paths": paths})
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"voices": [] if missing == "voice" else [{"voice_id": "saved-voice"}]})
        return httpx.Response(200, json={"data": []})
    mock_api(monkeypatch, handler)
    adapter = speech.VoiceStudio(config, config.speech_snapshot)
    monkeypatch.setattr(adapter, "launch", lambda: pytest.fail("must not launch over an existing API"))
    with pytest.raises(PipelineError, match={"routes": "incompatible", "voice": "profile not found", "model": "engine not installed"}[missing]):
        adapter.ensure_ready()


def test_existing_http_error_does_not_launch_duplicate_app(config, monkeypatch):
    mock_api(monkeypatch, lambda request: httpx.Response(503, json={"detail": "busy"}))
    adapter = speech.VoiceStudio(config, config.speech_snapshot)
    monkeypatch.setattr(adapter, "launch", lambda: pytest.fail("must not launch a duplicate app"))
    with pytest.raises(TemporaryError, match="503"):
        adapter.ensure_ready()


def test_mapping_preserves_numbers_unicode_offsets_and_token_splits():
    script = make_script("You're 38%!\nCafé.")
    tokens = [{"word": t, "start": i, "end": i+1} for i, t in enumerate([" You", "'re", " 38", "%!", " Café."])]
    data = speech.map_alignment(script, tokens, 5)
    words = aligned_words(script, data, 5)
    assert [(w.text, w.start, w.end) for w in words] == [("You're", 0, 2), ("38%!", 2, 4), ("Café.", 4, 5)]
    assert all(script.narration[w.char_start:w.char_end] == w.text for w in words)
    assert "character_start_times_seconds" not in data


@pytest.mark.parametrize("tokens", [[], [{"word": "different", "start": 0, "end": 1}],
    [{"word": "Hello.", "start": 1, "end": 1}], [{"word": "Hello.", "start": 0, "end": 11}],
    [{"word": "Hello.", "start": float("nan"), "end": 1}]])
def test_mapping_rejects_missing_mismatch_or_invalid_times(tokens):
    with pytest.raises(PipelineError):
        speech.map_alignment(make_script("Hello."), tokens, 10)


def test_alignment_retry_reuses_audio(config, tmp_path, local_speech, monkeypatch):
    pipeline = create(config, tmp_path)
    script = pipeline.script()
    monkeypatch.setattr(speech, "force_align", lambda *args: (_ for _ in ()).throw(PipelineError("alignment unavailable")))
    with pytest.raises(PipelineError, match="alignment unavailable"):
        pipeline.speech(script)
    source_hash = file_hash(pipeline.store.path("audio/source.wav"))
    monkeypatch.setattr(speech, "force_align", align_fixture)
    pipeline.speech(script)
    assert len(local_speech) == 1
    assert file_hash(pipeline.store.path("audio/source.wav")) == source_hash
    assert pipeline.store.manifest.usage == []


def test_alignment_version_refresh_preserves_completed_speech(config, tmp_path, local_speech, monkeypatch):
    pipeline = create(config, tmp_path)
    script = pipeline.script()
    alignments = []
    def align(*args):
        alignments.append(True)
        return align_fixture(*args)
    monkeypatch.setattr(speech, "force_align", align)
    pipeline.speech(script)
    source = file_hash(pipeline.store.path("audio/source.wav"))
    previous = pipeline.store.manifest.stages["speech.align"].input_hash
    monkeypatch.setattr(speech, "ALIGNMENT_VERSION", speech.ALIGNMENT_VERSION + 1)
    pipeline.speech(script)
    assert len(alignments) == 2 and len(local_speech) == 1
    assert pipeline.store.manifest.stages["speech.align"].input_hash != previous
    assert pipeline.store.manifest.stages["speech.chunk.0001"].attempts == 1
    assert file_hash(pipeline.store.path("audio/source.wav")) == source


@pytest.mark.parametrize("invalid_timing", [False, True, None])
def test_force_alignment_uses_configured_ffmpeg_and_cpu_pcm(config, tmp_path, monkeypatch, invalid_timing):
    audio = tmp_path / "source.wav"
    audio.write_bytes(wav_bytes())
    samples = object()
    calls = []
    class Array:
        def astype(self, dtype):
            assert dtype == "float32"
            return self
        def __truediv__(self, value):
            assert value == 32768.0
            return samples
    def frombuffer(data, dtype):
        assert dtype == "<i2" and len(data) == 8 * 16000 * 2
        return Array()
    class Model:
        def align(self, value, text, **options):
            assert value is samples and text == "Hello."
            # Punctuation splitting and postprocessing can collapse words in
            # otherwise valid long recordings. Return that failure if enabled.
            assert options["language"] == "en" and options["regroup"] is False
            assert options["failure_threshold"] == 0.0 and options["verbose"] is None
            end = 0 if invalid_timing is not False or options.get("presplit", True) or options.get("suppress_silence", True) else 1
            return types.SimpleNamespace(to_dict=lambda: {"segments": [{"words": [{"word": "Hello.", "start": 0, "end": end}]}]})
    def load_model(name, device):
        calls.append((name, device))
        return Model()
    monkeypatch.setitem(sys.modules, "stable_whisper", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=lambda threads: None))
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace(frombuffer=frombuffer, float32="float32"))
    from video_automation import ctc_alignment
    def refine(value, narration, tokens):
        assert value is samples and narration == "Hello."
        calls.append("refine")
        return [dict(token, end=1) for token in tokens] if invalid_timing is None else tokens
    monkeypatch.setattr(ctc_alignment, "refine_instant_words", refine)
    if invalid_timing:
        with pytest.raises(PipelineError, match="token 1 has invalid or zero-duration timing") as error:
            speech.force_align(config, audio, make_script("Hello."))
        assert "Local forced alignment failed (PipelineError)" not in str(error.value)
    else:
        result = speech.force_align(config, audio, make_script("Hello."))
        assert result["words"][0]["text"] == "Hello."
    assert calls == ([("base.en", "cpu")] if invalid_timing is False else [("base.en", "cpu"), "refine"])
    assert not (tmp_path / "alignment_input.tmp.wav").exists()


def test_partial_chunk_recovery(config, tmp_path, monkeypatch):
    pipeline = create(config, tmp_path, "First sentence. " * 400)
    calls = []
    audio = wav_bytes()
    def synthesize(self, text):
        calls.append(text)
        if len(calls) == 2:
            raise PipelineError("second chunk failed")
        return audio
    monkeypatch.setattr(speech.VoiceStudio, "synthesize", synthesize)
    monkeypatch.setattr(speech, "force_align", align_fixture)
    with pytest.raises(PipelineError, match="second chunk"):
        pipeline.speech(pipeline.script())
    pipeline.speech(pipeline.script())
    assert len(calls) == 3
    assert calls[0] != calls[1] and calls[1] == calls[2]


def test_saved_provider_identity_and_historical_preservation(config, tmp_path):
    pipeline = create(config, tmp_path)
    folder = pipeline.store.folder
    new_config = config.model_copy(update={"speech_provider": "elevenlabs", "voice_id": "other"})
    resumed = Pipeline(Store(folder, new_config))
    assert resumed.config.speech_provider == "voicestudio"
    assert resumed.config.voicestudio_voice == "saved-voice"
    source = tmp_path / "narration.txt"
    with pytest.raises(PipelineError, match="different speech settings"):
        Pipeline.create(new_config, script_path=source, job_id=pipeline.store.manifest.job_id)
    eleven = Pipeline.create(new_config, script_path=source)
    assert eleven.store.folder != folder
    manifest_path = eleven.store.path("manifest.json")
    legacy = read_json(manifest_path)
    legacy.pop("speech_settings")
    write_json(manifest_path, legacy)
    before = manifest_path.read_bytes()
    legacy_pipeline = Pipeline(Store(eleven.store.folder, config))
    assert legacy_pipeline.config.speech_provider == "elevenlabs"
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("pauses", [False, True])
def test_external_handoff_render_cached_resume_and_replacement(config, tmp_path, local_speech, monkeypatch, pauses):
    text = "You see 38% growth. " + ("[pause=1s] " if pauses else "") + "Keep going!"
    pipeline = create(config, tmp_path, text)
    result = pipeline.run()
    assert result["status"] == "waiting_for_images"
    assert "image" not in pipeline.providers.calls
    for cue in pipeline.store.manifest.cues:
        pipeline.store.path(f"images/{cue.id}.png").write_bytes(image_bytes())
    assert pipeline.run()["passed"]
    assert pipeline.store.manifest.status == "completed"
    audio_hash = file_hash(pipeline.store.path("audio/narration.wav"))
    monkeypatch.setattr(speech.VoiceStudio, "synthesize", lambda *a: pytest.fail("cached speech must not start app"))
    monkeypatch.setattr(speech, "force_align", lambda *a: pytest.fail("cached alignment must not load model"))
    assert pipeline.run()["passed"]
    assert len(local_speech) == 1
    assert file_hash(pipeline.store.path("audio/narration.wav")) == audio_hash
    assert media.validate(pipeline.store)["passed"]
    previous_video = file_hash(pipeline.store.path("final.mp4"))
    cue = pipeline.store.manifest.cues[0]
    Image.new("RGB", (640, 360), "#118844").save(pipeline.store.path(f"images/{cue.id}.png"))
    assert pipeline.run()["passed"]
    assert file_hash(pipeline.store.path("final.mp4")) != previous_video
    assert file_hash(pipeline.store.path("audio/narration.wav")) == audio_hash
    data = read_json(pipeline.store.path("audio/word_alignment.json"))
    data["words"][0]["text"] = "corrupt"
    write_json(pipeline.store.path("audio/word_alignment.json"), data)
    with pytest.raises(PipelineError, match="Final validation failed"):
        media.validate(pipeline.store)


def test_voice_settings_change_creates_new_job(config, tmp_path):
    first = create(config, tmp_path)
    second = create(config.model_copy(update={"voicestudio_voice": "another-voice"}), tmp_path)
    assert first.store.folder != second.store.folder


def test_doctor_is_read_only_and_does_not_require_elevenlabs(config, monkeypatch, capsys):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config.voicestudio_auto_launch = False
    monkeypatch.setattr(media, "run", lambda *args: "present")
    monkeypatch.setattr(speech.VoiceStudio, "inspect", lambda self: True)
    monkeypatch.setattr(speech.VoiceStudio, "launch", lambda *args: pytest.fail("doctor must not launch"))
    monkeypatch.setattr(speech, "force_align", lambda *args: pytest.fail("doctor must not align"))
    monkeypatch.setattr(speech, "alignment_readiness", lambda: {"dependency": True, "model_cached": True})
    assert doctor(config) == 0
    report = json.loads(capsys.readouterr().out)
    assert "ELEVENLABS_API_KEY" not in report["checks"]
    assert report["speech_provider"] == "voicestudio"


def test_cli_external_handoff_without_elevenlabs(config, tmp_path, local_speech, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    write_json(config_path, config.model_dump(mode="json"))
    source = tmp_path / "text.txt"
    source.write_text("You see 38% growth.", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(Providers, "metadata", lambda *args: {"title": "Growth", "description": "A growth example."})
    monkeypatch.setattr(Providers, "visuals", FakeProviders().visuals)
    assert main(["--config", str(config_path), "run", "--script", str(source), "--no-thumbnail"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "waiting_for_images"
