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
from .providers import AUDIT_PROMPT, CHARACTERS, RESEARCH_PROMPT, SCRIPT_PROMPT, STYLE, VISUAL_PROMPT, Providers
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
        job_id = job_id or "video_" + digest(identity)[:12]
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
                                                  script_input_hash=script_input_hash, fps=config.fps))
        if script is not None:
            atomic_bytes(store.path("input_script.txt"), script.narration.encode("utf-8"))
        return cls(store)

    def json_task(self, name, inputs, relative, action):
        def save():
            data = action()
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
        # Bounded batches prevent an entire long video from exceeding structured-output limits.
        records = []
        for offset in range(0, len(cues), 24):
            batch = cues[offset:offset+24]
            previous_scenes = records[-3:]
            planned = self.json_task(f"visuals.{offset//24:03}", {"cues": [c.model_dump() for c in batch],
                                     "previous_scenes": previous_scenes,
                                     "text_generation": self.config.text_cache_settings, "style": STYLE, "prompt": VISUAL_PROMPT},
                                     f"visuals/batch_{offset//24:03}.json",
                                     lambda batch=batch, previous_scenes=previous_scenes: self.providers.visuals(batch, previous_scenes))
            records.extend(planned["cues"])
        if [r["id"] for r in records] != [c.id for c in cues]:
            raise PipelineError("Visual plan must cover every cue ID exactly once in order")
        for cue, record in zip(cues, records):
            for field in ("scene_group", "character_ids", "scene_description", "overlay_text", "motion"):
                setattr(cue, field, record[field])
            Cue.model_validate(cue.model_dump())
            if not cue.scene_group.strip() or not cue.scene_description.strip():
                raise PipelineError(f"Empty visual description or scene group: {cue.id}")
            if not set(cue.character_ids) <= {"parent", "infant", "caregiver"}:
                raise PipelineError(f"Unknown shared character ID: {cue.id}")
            if cue.overlay_text:
                cue.motion = "hold"
        manifest.cues = cues
        write_json(store.path("transcript.json"), [c.model_dump() for c in cues])
        store.save()

    def generate_image(self, prompt, references, destination):
        # The provider result is validated before it becomes the approved checkpoint.
        content = self.providers.image(prompt, references)
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
        store, manifest = self.store, self.store.manifest
        custom_reference = self.config.character_reference
        reference_key = digest({"characters": CHARACTERS, "style": STYLE, "image_generation": self.config.image_cache_settings})
        shared = self.config.jobs_dir / "references" / f"characters_{reference_key[:16]}.png"
        reference_inputs = {"key": reference_key, "custom": file_hash(custom_reference) if custom_reference else None}

        def reference():
            target = store.path("images/character_reference.png")
            if custom_reference:
                with Image.open(custom_reference) as image:
                    buffer = io.BytesIO()
                    image.convert("RGB").save(buffer, format="PNG")
                    atomic_bytes(target, buffer.getvalue())
                media.check_image(target)
            else:
                try:
                    media.check_image(shared)
                except PipelineError:
                    self.generate_image(CHARACTERS, [], shared)
                atomic_bytes(target, shared.read_bytes())
            return {"files": ["images/character_reference.png"]}

        store.task("image.character_reference", reference_inputs, reference)
        character = store.path("images/character_reference.png")
        group_references = {}
        for cue in manifest.cues:
            relative = f"images/{cue.id}.png"
            references = [character]
            if cue.scene_group in group_references:
                references.append(group_references[cue.scene_group])
            prompt = (f"Characters in this scene: {', '.join(cue.character_ids) or 'none'}. "
                      f"Scene group: {cue.scene_group}. {cue.scene_description}")
            inputs = {"prompt": prompt, "style": STYLE, "image_generation": self.config.image_cache_settings,
                      "references": [file_hash(p) for p in references]}

            def generate(relative=relative, references=references, prompt=prompt):
                result = self.generate_image(prompt, references, store.path(relative))
                return {"files": [relative], "validation": result}

            store.task(f"image.{cue.id}", inputs, generate)
            cue.image_path = relative
            group_references.setdefault(cue.scene_group, store.path(relative))
            if cue.overlay_text:
                overlay_relative = f"overlays/{cue.id}.png"
                def labels(cue=cue, overlay_relative=overlay_relative):
                    result = media.overlay(self.config, cue.overlay_text, store.path(overlay_relative))
                    return {"files": [overlay_relative], "layout": result}
                store.task(f"overlay.{cue.id}", {"text": cue.overlay_text, "width": self.config.width,
                           "height": self.config.height, "font": file_hash(self.config.font_path) if self.config.font_path else "system-default-v1"}, labels)
                cue.overlay_path = overlay_relative
            store.save()

    def run(self):
        store, manifest = self.store, self.store.manifest
        manifest.status, manifest.error = "running", None
        store.save()
        try:
            if manifest.mode in {"production", "script"}:
                script = self.script()
                words = self.speech(script)
                self.visuals(words)
                self.images()
            media.render(store)
            media.finish(store)
            report = media.validate(store)
            manifest.status = "completed"
            store.save()
            return report
        except KeyboardInterrupt:
            manifest.status, manifest.error = "interrupted", "Interrupted; completed checkpoints are preserved"
            store.save()
            raise
        except Exception as exc:
            manifest.status, manifest.error = "failed", str(exc)
            store.save()
            write_json(store.path("failure_report.json"), {"error": str(exc), "stage_status": {
                k: {"status": v.status, "attempts": v.attempts, "errors": v.errors} for k, v in manifest.stages.items()}})
            raise
