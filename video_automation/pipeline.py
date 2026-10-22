from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path

from PIL import Image

from . import media
from .models import Cue, Manifest, Script
from .progress import WorkflowProgress
from .providers import AUDIT_PROMPT, RESEARCH_PROMPT, SCRIPT_PROMPT, STYLE, VISUAL_PROMPT, Providers, validate_visual_plan
from .store import PipelineError, Store, TemporaryError, atomic_bytes, digest, file_hash, read_json, write_json
from .timing import aligned_words, make_script, normalize, segment_words


class Pipeline:
    def __init__(self, store: Store, providers=None):
        self.store = store
        self.config = store.config
        self.providers = providers or Providers(store)

    @classmethod
    def create(cls, config, topic: str | None = None, job_id: str | None = None, *, script_path: Path | None = None):
        script = make_script(script_path.read_text(encoding="utf-8-sig")) if script_path is not None else None
        mode = "script" if script is not None else "production"
        script_input_hash = digest(script.narration) if script is not None else None
        topic = (topic or (script_path.stem if script_path is not None else "")).strip()
        if not topic:
            raise PipelineError("Topic cannot be empty")
        identity = {"mode": mode, "narration": script.narration} if script is not None else topic
        job_id = job_id or "video_native_text_" + digest(identity)[:12]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job_id):
            raise PipelineError("Job ID must contain only letters, digits, underscores and hyphens")
        folder = config.jobs_dir / job_id
        if (folder / "manifest.json").is_file():
            store = Store(folder, config)
            if (store.manifest.mode != mode
                    or (script is None and store.manifest.topic != topic)
                    or (script is not None and store.manifest.script_input_hash != script_input_hash)):
                raise PipelineError("Existing job has a different topic, script, or mode; choose another job ID")
        else:
            store = Store(folder, config, Manifest(job_id=job_id, topic=topic, mode=mode,
                                                  image_strategy="native_text", script_input_hash=script_input_hash, fps=config.fps))
        if script is not None:
            atomic_bytes(store.path("input_script.txt"), script.narration.encode("utf-8"))
        return cls(store)

    def json_task(self, name, inputs, relative, action, validate=None):
        previous = self.store.manifest.stages.get(name)
        if validate and previous and previous.status == "completed" and previous.input_hash == digest(inputs):
            try:
                validate(previous.result.get("data"))
            except TemporaryError as exc:
                # Older checkpoints may have been saved before semantic validation existed.
                previous.status = "failed"
                previous.errors.append(str(exc))
                self.store.save()
                self.store.log("cache_rejected", stage=name, error=str(exc))
        def save():
            data = action()
            if validate:
                validate(data)
            write_json(self.store.path(relative), data)
            return {"files": [relative], "data": data}
        return self.store.task(name, inputs, save)["data"]

    def script(self) -> Script:
        store, manifest = self.store, self.store.manifest
        if manifest.mode == "script":
            narration = store.path("input_script.txt").read_text(encoding="utf-8")
            if digest(narration) != manifest.script_input_hash:
                raise PipelineError("Saved input script hash mismatch; rerun with the original narration file to restore it")
            return self.save_script(make_script(narration))
        research = self.json_task("research", {"topic": manifest.topic, "text_generation": self.config.text_cache_settings, "prompt": RESEARCH_PROMPT},
                                  "sources.json", lambda: self.providers.research(manifest.topic))
        draft = self.json_task("script.draft", {"research": research, "topic": manifest.topic,
                              "text_generation": self.config.text_cache_settings, "prompt": SCRIPT_PROMPT},
                              "draft.json", lambda: self.providers.script(manifest.topic, research))
        if not 4500 <= len(normalize(draft["narration"])) <= 5000:
            draft = self.json_task("script.revision", {"draft": draft, "research": research, "prompt": SCRIPT_PROMPT,
                                  "topic": manifest.topic, "text_generation": self.config.text_cache_settings},
                                  "draft_revision.json", lambda: self.providers.script(manifest.topic, research, draft))
        script = make_script(draft["narration"], draft["source_references"])
        allowed = {s["url"] for s in research["sources"]}
        if not script.source_references or not set(script.source_references) <= allowed:
            raise PipelineError("Script source references are missing or not present in research")

        def audit():
            result = self.providers.audit(script, research)
            records = result.get("sentences", [])
            if sorted(r["sentence_id"] for r in records) != sorted(s.id for s in script.sentences):
                raise PipelineError("Source audit did not cover every sentence exactly once")
            factual_kinds = {"observation", "hypothesis", "modern_connection"}
            for record in records:
                if (not record["supported"] or not set(record["source_urls"]) <= allowed
                        or (record["kind"] in factual_kinds and not record["source_urls"])):
                    write_json(store.path("source_audit_failed.json"), result)
                    raise PipelineError(f"Unsupported narration sentence {record['sentence_id']}: {record['reason']}")
            return result

        audit_result = self.json_task("source_audit", {"script": script.model_dump(), "research": research,
                       "text_generation": self.config.text_cache_settings, "prompt": AUDIT_PROMPT}, "source_audit.json", audit)
        script.source_references = sorted(set(script.source_references) | {
            url for record in audit_result["sentences"] for url in record["source_urls"]})
        return self.save_script(script)

    def save_script(self, script: Script) -> Script:
        store = self.store
        atomic_bytes(store.path("script.txt"), script.narration.encode("utf-8"))
        write_json(store.path("script.json"), script.model_dump(mode="json"))
        store.manifest.script_character_count = script.character_count
        store.save()
        return script

    def speech(self, script: Script):
        store, manifest = self.store, self.store.manifest
        # Repeat the invariant at the provider boundary, including resumed jobs.
        if script.narration != normalize(script.narration) or not 4500 <= len(script.narration) <= 5000:
            raise PipelineError("Speech blocked: normalized script must contain 4500–5000 characters")
        speech = self.json_task("speech.request", {"text": script.narration, "model": self.config.elevenlabs_model,
                                "voice": self.config.voice_id, "settings": self.config.voice_settings},
                                "audio/speech_response.json", lambda: self.providers.speech(script.narration))

        def decode():
            try:
                encoded = base64.b64decode(speech["audio_base64"], validate=True)
            except (ValueError, KeyError) as exc:
                raise PipelineError("Speech provider returned invalid audio") from exc
            atomic_bytes(store.path("audio/provider.mp3"), encoded)
            audio = store.path("audio/narration.wav")
            temporary = audio.with_suffix(".tmp.wav")
            media.ffmpeg(self.config, "-loglevel", "error", "-i", store.path("audio/provider.mp3"),
                         "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", temporary)
            temporary.replace(audio)
            duration = media.audio_duration(self.config, audio)
            alignment = speech.get("alignment")
            if not alignment:
                raise PipelineError("Speech provider returned no original character alignment; realignment is required")
            words = aligned_words(script, alignment, duration)
            sentences = [{"id": sentence.id, "text": sentence.text,
                          "start": min(w.start for w in words if w.sentence_id == sentence.id),
                          "end": max(w.end for w in words if w.sentence_id == sentence.id)} for sentence in script.sentences]
            write_json(store.path("alignment.json"), {"characters": alignment, "words": [w.model_dump() for w in words],
                       "sentences": sentences, "audio_duration": duration, "audio_sha256": file_hash(audio)})
            return {"files": ["audio/narration.wav", "alignment.json"], "duration": duration}

        result = store.task("speech.decode", {"speech": digest(speech), "script": script.model_dump(), "decode_version": 1}, decode)
        manifest.audio_path = "audio/narration.wav"
        manifest.audio_duration = result["duration"]
        store.save()
        from .models import Word
        return [Word.model_validate(w) for w in read_json(store.path("alignment.json"))["words"]]

    def visuals(self, words):
        store, manifest = self.store, self.store.manifest
        cues = segment_words(words, self.config.fps, manifest.audio_duration)
        if store.progress is not None:
            store.progress.set_total((len(cues) + 23) // 24, "batches")
        # Bounded batches prevent an entire long video from exceeding structured-output limits.
        records = []
        for offset in range(0, len(cues), 24):
            batch = cues[offset:offset+24]
            previous_scenes = records[-3:]
            planned = self.json_task(f"visuals.{offset//24:03}", {"cues": [c.model_dump() for c in batch],
                                     "previous_scenes": previous_scenes,
                                     "text_generation": self.config.text_cache_settings, "style": STYLE, "prompt": VISUAL_PROMPT},
                                     f"visuals/batch_{offset//24:03}.json",
                                     lambda batch=batch, previous_scenes=previous_scenes: self.providers.visuals(batch, previous_scenes),
                                     validate=lambda planned, batch=batch: validate_visual_plan(batch, planned))
            records.extend(planned["cues"])
        if [r["id"] for r in records] != [c.id for c in cues]:
            raise PipelineError("Visual plan must cover every cue ID exactly once in order")
        for cue, record in zip(cues, records):
            for field in ("scene_group", "character_ids", "scene_description", "motion"):
                setattr(cue, field, record[field])
            Cue.model_validate(cue.model_dump())
        manifest.cues = cues
        write_json(store.path("transcript.json"), [c.model_dump() for c in cues])
        store.save()

    def generate_image(self, prompt, destination):
        # The provider result is validated before it becomes the approved checkpoint.
        content = self.providers.image(prompt)
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                output = io.BytesIO()
                image.convert("RGB").save(output, format="PNG")
            atomic_bytes(destination, output.getvalue())
            result = media.check_image(destination)
        except (OSError, ValueError, PipelineError) as exc:
            raise TemporaryError(f"Generated image failed validation: {exc}") from exc
        if self.config.image_interval_seconds:
            import time
            time.sleep(self.config.image_interval_seconds)
        return result

    def images(self):
        """Checkpoint all standalone prompts before generating one image per cue."""
        store, manifest = self.store, self.store.manifest
        if (not manifest.cues or len({cue.id for cue in manifest.cues}) != len(manifest.cues)
                or any(not cue.scene_description.strip() for cue in manifest.cues)):
            raise PipelineError("Image prompts require unique cues with nonblank scene descriptions")
        expected = [{"cue_id": cue.id, "prompt": cue.scene_description.strip() + "\n" + STYLE}
                    for cue in manifest.cues]

        def validate_prompts(records):
            if records != expected:
                raise TemporaryError("Image prompts must match every cue exactly once in order with its complete prompt")

        prompts = self.json_task("image.prompts", {"prompts": expected, "prompt_version": 2},
                                 "image_prompts.json", lambda: expected, validate=validate_prompts)
        for cue, record in zip(manifest.cues, prompts):
            relative = f"images/{cue.id}.png"
            prompt = record["prompt"]
            inputs = {"prompt": prompt, "image_generation": self.config.image_cache_settings}
            previous = manifest.stages.get(f"image.{cue.id}")
            if previous and "gemini_fallback_model" in inputs["image_generation"]:
                legacy_settings = {key: value for key, value in inputs["image_generation"].items()
                                   if key != "gemini_fallback_model"}
                if previous.input_hash == digest({"prompt": prompt, "image_generation": legacy_settings}):
                    # Existing primary images and retry budgets remain valid when adding fallback.
                    previous.input_hash = digest(inputs)

            def generate(relative=relative, prompt=prompt):
                result = self.generate_image(prompt, store.path(relative))
                return {"files": [relative], "validation": result}

            store.task(f"image.{cue.id}", inputs, generate)
            cue.image_path = relative
            store.save()

    def run(self):
        store, manifest = self.store, self.store.manifest
        manifest.status, manifest.error = "running", None
        store.save()
        phases = (["narration", "speech", "visuals", "images"] if manifest.mode in {"production", "script"} else [])
        progress = WorkflowProgress(store, phases + ["render", "finish", "validate"])
        store.progress = progress
        try:
            if manifest.mode in {"production", "script"}:
                with progress.phase("narration"):
                    script = self.script()
                with progress.phase("speech"):
                    words = self.speech(script)
                with progress.phase("visuals"):
                    self.visuals(words)
                with progress.phase("images"):
                    self.images()
            with progress.phase("render"):
                media.render(store)
            with progress.phase("finish"):
                media.finish(store)
            with progress.phase("validate"):
                report = media.validate(store)
            manifest.status = "completed"
            store.save()
            progress.finish("completed")
            return report
        except KeyboardInterrupt:
            manifest.status, manifest.error = "interrupted", "Interrupted; completed checkpoints are preserved"
            store.save()
            progress.finish("interrupted", manifest.error)
            raise
        except Exception as exc:
            manifest.status, manifest.error = "failed", str(exc)
            store.save()
            write_json(store.path("failure_report.json"), {"error": str(exc), "stage_status": {
                k: {"status": v.status, "attempts": v.attempts, "errors": v.errors} for k, v in manifest.stages.items()}})
            progress.finish("failed", manifest.error)
            raise
        finally:
            store.progress = None
