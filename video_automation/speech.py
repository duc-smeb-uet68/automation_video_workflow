"""Local VoiceStudio transport and forced alignment of unchanged narration."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import re
import subprocess
import time
import wave
from pathlib import Path

import httpx

from . import media
from .models import SpeechSettings, Word
from .providers import Providers
from .store import PipelineError, TemporaryError, atomic_bytes, digest, file_hash, read_json, write_json
from .timing import aligned_words, pause_timing, SPEECH_SAMPLE_RATE


ALIGNMENT_MODEL = "base.en"
ALIGNMENT_VERSION = 1


class BackendUnavailable(TemporaryError):
    """The loopback connection failed before receiving an HTTP response."""


def speech_chunks(text: str, limit: int = 4096) -> list[str]:
    """Partition text without removing whitespace or changing its content."""
    chunks = []
    while len(text) > limit:
        window = text[:limit]
        boundaries = list(re.finditer(r"[.!?][\"'\u201d\u2019]*(?:\s+)", window))
        spaces = list(re.finditer(r"\s+", window))
        end = boundaries[-1].end() if boundaries else spaces[-1].end() if spaces else limit
        chunks.append(text[:end])
        text = text[end:]
    if text:
        chunks.append(text)
    return chunks


def alignment_readiness() -> dict:
    """Inspect installed dependencies and model cache without loading weights."""
    available = importlib.util.find_spec("stable_whisper") is not None
    cached = False
    if available:
        # Importing Whisper does not load or download a model.
        try:
            import stable_whisper
            import whisper
            url = whisper._MODELS[ALIGNMENT_MODEL]
            root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
            model = root / url.rsplit("/", 1)[-1]
            if model.is_file():
                expected = url.split("/")[-2]
                with model.open("rb") as stream:
                    cached = hashlib.file_digest(stream, "sha256").hexdigest() == expected
        except (ImportError, OSError, KeyError, RuntimeError):
            available = False
    return {"dependency": available, "model_cached": cached, "model": ALIGNMENT_MODEL, "device": "cpu"}


class VoiceStudio:
    def __init__(self, config, settings: SpeechSettings, store=None):
        self.config, self.settings, self.store = config, settings, store
        self.ready = False

    def announce(self, message):
        if self.store is not None and self.store.progress is not None:
            self.store.progress.console(message)

    def request(self, method, route, payload=None, timeout=5):
        headers = {}
        if key := os.environ.get("VOICESTUDIO_API_KEY"):
            headers["Authorization"] = "Bearer " + key
        try:
            with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False) as client:
                response = client.request(method, self.settings.base_url.rstrip("/") + route,
                                          headers=headers, json=payload)
        except httpx.TransportError as exc:
            raise BackendUnavailable(f"VoiceStudio API unavailable: {type(exc).__name__}") from exc
        if not response.is_success:
            details = Providers.error_details(response, headers)
            reason = details.get("message") or details.get("detail") or "Check VoiceStudio model installation and API settings"
            error = TemporaryError if response.status_code in {408, 409, 425, 429} or response.status_code >= 500 else PipelineError
            raise error(f"VoiceStudio HTTP {response.status_code}: {reason}")
        return response

    def inspect(self):
        """Read only API metadata; do not start the application or load engines."""
        try:
            spec = self.request("GET", "/openapi.json").json()
            required = {"/v1/audio/speech", "/v1/audio/voices", "/v1/models"}
            if not required.issubset(spec.get("paths", {})):
                raise PipelineError("VoiceStudio API is incompatible: required audio routes are missing")
            voices = self.request("GET", "/v1/audio/voices").json()
            voices = voices.get("voices", voices.get("data", [])) if isinstance(voices, dict) else voices
            if self.settings.voice != "default" and not any(v.get("voice_id") == self.settings.voice for v in voices):
                raise PipelineError(f"VoiceStudio voice profile not found: {self.settings.voice}")
            models = self.request("GET", "/v1/models").json()
            if not any(m.get("id") == self.settings.model for m in models.get("data", [])):
                raise PipelineError(f"VoiceStudio engine not installed or unavailable: {self.settings.model}. Install it in the app.")
        except (ValueError, TypeError, AttributeError) as exc:
            raise PipelineError("VoiceStudio API returned incompatible metadata") from exc
        return True

    def launch(self):
        executable = self.settings.executable
        if not executable or not executable.is_file():
            raise PipelineError("Set voicestudio_executable to the installed VoiceStudio application")
        options = {}
        if os.name == "nt":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            options["startupinfo"] = startup
        try:
            subprocess.Popen([str(executable)], cwd=str(executable.parent), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
        except OSError as exc:
            raise PipelineError(f"Could not launch VoiceStudio: {type(exc).__name__}") from exc

    def ensure_ready(self):
        if self.ready:
            return
        try:
            self.inspect()
        except BackendUnavailable:
            if not self.settings.auto_launch:
                raise PipelineError("VoiceStudio API unavailable. Open the installed app and resume the job.") from None
            self.announce("Starting VoiceStudio; waiting up to 180 seconds for its local API.")
            self.launch()
            deadline, notice = time.monotonic() + 180, time.monotonic() + 15
            while True:
                try:
                    self.inspect()
                    break
                except TemporaryError:
                    if time.monotonic() >= deadline:
                        raise TemporaryError("VoiceStudio startup timed out after 180 seconds. Check the installed app and resume.") from None
                    if time.monotonic() >= notice:
                        self.announce("Waiting for VoiceStudio backend startup...")
                        notice = time.monotonic() + 15
                    time.sleep(1)
        self.ready = True

    def synthesize(self, text: str) -> bytes:
        self.ensure_ready()
        payload = {"model": self.settings.model, "voice": self.settings.voice, "input": text,
                   "response_format": "wav", "language": self.settings.language, "speed": self.settings.speed}
        response = self.request("POST", "/v1/audio/speech", payload, self.config.request_timeout_seconds)
        try:
            with wave.open(io.BytesIO(response.content), "rb") as audio:
                if audio.getnframes() <= 0 or audio.getcomptype() != "NONE":
                    raise ValueError("empty or compressed WAV")
                expected = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
                if len(audio.readframes(audio.getnframes())) != expected:
                    raise ValueError("truncated WAV")
        except (wave.Error, EOFError, ValueError) as exc:
            raise TemporaryError("VoiceStudio returned invalid or empty WAV audio") from exc
        return response.content


def map_alignment(script, tokens: list[dict], duration: float) -> dict:
    """Map actual aligned token spans to the exact original whitespace words."""
    expected = "".join(script.narration.split())
    spans, offset = [], 0
    try:
        for token in tokens:
            text = "".join(token["word"].split())
            if not text:
                continue
            start, end = float(token["start"]), float(token["end"])
            if not (0 <= start < end <= duration) or (spans and start < spans[-1][3] - 0.00001):
                raise ValueError("invalid token timing")
            spans.append((offset, offset + len(text), start, end, text))
            offset += len(text)
        if "".join(t[4] for t in spans) != expected:
            raise ValueError("text mismatch")
        words, offset, cursor = [], 0, 0
        for match in re.finditer(r"\S+", script.narration):
            last = offset + len(match.group())
            first = cursor
            while cursor < len(spans) and spans[cursor][1] <= last:
                cursor += 1
            if first == cursor or spans[first][0] != offset or spans[cursor-1][1] != last:
                raise ValueError("token crosses original word boundary")
            sentence = next(s for s in script.sentences if s.char_start <= match.start() < s.char_end)
            words.append(Word(text=match.group(), start=spans[first][2], end=spans[cursor-1][3],
                              char_start=match.start(), char_end=match.end(), sentence_id=sentence.id).model_dump())
            offset = last
    except (KeyError, TypeError, ValueError, StopIteration, IndexError) as exc:
        raise PipelineError("Forced alignment failed exact text coverage or valid word timing; generated audio is preserved") from exc
    result = {"format": "word_alignment_v1", "narration": script.narration, "words": words}
    aligned_words(script, result, duration)
    return result


def force_align(config, audio: Path, script) -> dict:
    try:
        import stable_whisper
        import torch
        import numpy as np
    except ImportError as exc:
        raise PipelineError('VoiceStudio alignment requires: python -m pip install -e ".[voicestudio]"') from exc
    torch.set_num_threads(config.threads)
    try:
        # Pass PCM directly so Stable-ts never assumes FFmpeg is on PATH.
        decoded = audio.with_name("alignment_input.tmp.wav")
        try:
            media.ffmpeg(config, "-loglevel", "error", "-i", audio, "-vn", "-ac", "1", "-ar", "16000",
                         "-c:a", "pcm_s16le", decoded)
            with wave.open(str(decoded), "rb") as stream:
                samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        finally:
            decoded.unlink(missing_ok=True)
        model = stable_whisper.load_model(ALIGNMENT_MODEL, device="cpu")
        result = model.align(samples, script.narration, language="en", regroup=False,
                             failure_threshold=0.0, verbose=None)
        if result is None:
            raise PipelineError("Forced alignment failed; generated audio is preserved")
        tokens = [word for segment in result.to_dict()["segments"] for word in segment["words"]]
        return map_alignment(script, tokens, media.audio_duration(config, audio))
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        raise PipelineError(f"Local forced alignment failed ({type(exc).__name__}); generated audio is preserved") from exc


def voicestudio_speech(store, script):
    config, settings = store.config, store.manifest.speech_settings
    adapter = VoiceStudio(config, settings, store)
    chunks, records = speech_chunks(script.narration), []
    snapshot = settings.model_dump(mode="json")
    for index, text in enumerate(chunks, 1):
        relative = f"audio/voicestudio/chunk_{index:04}.wav"
        def generate(text=text, relative=relative):
            atomic_bytes(store.path(relative), adapter.synthesize(text))
            store.log("local_speech_received", chunk=index, characters=len(text), provider="voicestudio")
            return {"files": [relative]}
        store.task(f"speech.chunk.{index:04}", {"text": text, "settings": snapshot}, generate)
        records.append({"path": relative, "sha256": file_hash(store.path(relative))})

    def assemble():
        output = store.path("audio/source.wav")
        temporary = output.with_suffix(".tmp.wav")
        with wave.open(str(temporary), "wb") as joined:
            joined.setparams((1, 2, SPEECH_SAMPLE_RATE, 0, "NONE", "not compressed"))
            for record in records:
                part = store.path("audio/voicestudio/decoded.tmp.wav")
                media.ffmpeg(config, "-loglevel", "error", "-i", store.path(record["path"]),
                             "-vn", "-ac", "1", "-ar", SPEECH_SAMPLE_RATE, "-c:a", "pcm_s16le", part)
                with wave.open(str(part), "rb") as audio:
                    while data := audio.readframes(65536):
                        joined.writeframesraw(data)
                part.unlink()
        media.audio_duration(config, temporary)
        temporary.replace(output)
        return {"files": ["audio/source.wav"]}
    store.task("speech.assemble", {"chunks": records, "version": 1}, assemble)
    source = store.path("audio/source.wav")

    def align():
        adapter.announce("Aligning original narration on CPU; first use downloads the base.en alignment model.")
        data = force_align(config, source, script)
        data["source_audio_sha256"] = file_hash(source)
        write_json(store.path("audio/word_alignment.json"), data)
        return {"files": ["audio/word_alignment.json"]}
    store.task("speech.align", {"audio": file_hash(source), "narration": script.narration,
                               "model": ALIGNMENT_MODEL, "version": ALIGNMENT_VERSION}, align)
    original = read_json(store.path("audio/word_alignment.json"))
    aligned_words(script, original, media.audio_duration(config, source))

    def finish():
        alignment = original
        temporary = store.path("audio/narration.tmp.wav")
        atomic_bytes(temporary, source.read_bytes())
        with wave.open(str(source), "rb") as audio:
            frames = audio.getnframes()
        pauses = []
        if store.manifest.script_pauses:
            alignment, pauses = pause_timing(script, original, frames, store.manifest.script_pauses)
            media.insert_silence(temporary, pauses)
        duration = media.audio_duration(config, temporary)
        words = aligned_words(script, alignment, duration)
        output = store.path("audio/narration.wav")
        temporary.replace(output)
        sentences = [{"id": s.id, "text": s.text,
                      "start": min(w.start for w in words if w.sentence_id == s.id),
                      "end": max(w.end for w in words if w.sentence_id == s.id)} for s in script.sentences]
        report = {"format": "word_alignment_v1", "source_alignment": alignment,
                  "words": [w.model_dump() for w in words], "sentences": sentences,
                  "audio_duration": duration, "audio_sha256": file_hash(output)}
        if pauses:
            report["pauses"] = pauses
        write_json(store.path("alignment.json"), report)
        return {"files": ["audio/narration.wav", "alignment.json"], "duration": duration}
    result = store.task("speech.finish", {"audio": file_hash(source), "alignment": digest(original),
                        "pauses": [p.model_dump() for p in store.manifest.script_pauses], "version": 1}, finish)
    store.manifest.audio_path, store.manifest.audio_duration = "audio/narration.wav", result["duration"]
    store.save()
    return [Word.model_validate(w) for w in read_json(store.path("alignment.json"))["words"]]
