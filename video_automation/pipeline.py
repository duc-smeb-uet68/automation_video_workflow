from __future__ import annotations

import base64
import io
import json
import math
import re
import wave
from pathlib import Path

from PIL import Image

from . import media
from .models import Cue, Manifest, Script
from .progress import WorkflowProgress
from .providers import (AUDIT_PROMPT, CUE_SPLIT_CONTRACT, CUE_SPLIT_PROMPT, CUE_SPLIT_SCHEMA,
                        METADATA_PROMPT, METADATA_SCHEMA, RESEARCH_PROMPT, SCRIPT_PROMPT,
                        STYLE, THUMBNAIL_LAYOUT, THUMBNAIL_PROMPT, THUMBNAIL_SCHEMA, VISUAL_CONTEXT_PROMPT,
                        VISUAL_CONTEXT_SCHEMA, VISUAL_PROMPT, VISUAL_RESPONSE_VERSION, Providers, scene_prompt, thumbnail_prompt,
                        validate_cue_plan, validate_metadata, validate_thumbnail, validate_visual_context, validate_visual_plan)
from .store import PipelineError, Store, TemporaryError, atomic_bytes, digest, file_hash, read_json, write_json
from .timing import SPEECH_SAMPLE_RATE, aligned_words, make_script, normalize, parse_script_input, pause_timing, segment_words, subtitles


VISUAL_BATCH_SIZE = 8


class ImagesPending(Exception):
    """A normal handoff to the user, with no failed stage or retry consumed."""

    def __init__(self, report):
        self.report = report


class Pipeline:
    def __init__(self, store: Store, providers=None):
        self.store = store
        self.config = store.config
        if store.manifest.mode in {"script", "production"}:
            self.config = store.config = store.config.with_speech(store.manifest.speech_settings)
        self.providers = providers or Providers(store)

    @classmethod
    def create(cls, config, topic: str | None = None, job_id: str | None = None, *,
               script_path: Path | None = None, thumbnail_enabled: bool | None = None):
        script_input, pauses = None, []
        if script_path is not None:
            script_input = normalize(script_path.read_text(encoding="utf-8-sig"))
            narration, pauses = parse_script_input(script_input)
            script = make_script(narration)
        else:
            script = None
        mode = "script" if script is not None else "production"
        script_input_hash = digest(script_input) if script is not None else None
        topic = (topic or (script_path.stem if script_path is not None else "")).strip()
        if not topic:
            raise PipelineError("Topic cannot be empty")
        identity = {"mode": mode, "narration": script_input} if script is not None else topic
        if config.image_mode == "external":
            identity = {"input": identity, "image_mode": "external"}
        speech_settings = config.speech_snapshot
        if speech_settings.provider == "voicestudio":
            identity = {"input": identity, "speech": speech_settings.model_dump(mode="json")}
        job_id = job_id or "video_native_text_" + digest(identity)[:12]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job_id):
            raise PipelineError("Job ID must contain only letters, digits, underscores and hyphens")
        folder = config.jobs_dir / job_id
        if (folder / "manifest.json").is_file():
            store = Store(folder, config)
            saved_speech = store.manifest.speech_settings
            if ((saved_speech.provider if saved_speech else "elevenlabs") != speech_settings.provider
                    or (saved_speech is not None and saved_speech != speech_settings)):
                raise PipelineError("Existing job has different speech settings; choose another job ID. Resume uses saved speech settings.")
            if (store.manifest.mode != mode
                    or store.manifest.image_mode != config.image_mode
                    or (script is None and store.manifest.topic != topic)
                    or (script is not None and store.manifest.script_input_hash != script_input_hash)):
                raise PipelineError("Existing job has a different topic, script, or mode; choose another job ID. "
                                    "Use resume --image-mode to change an existing job's image mode explicitly.")
            if thumbnail_enabled is not None and thumbnail_enabled != store.manifest.thumbnail_enabled:
                raise PipelineError("Existing job has a different thumbnail selection; use resume --thumbnail "
                                    "or --no-thumbnail to change it explicitly.")
            if pauses != store.manifest.script_pauses:
                raise PipelineError("Existing job has different pause instructions; choose another job ID")
        else:
            store = Store(folder, config, Manifest(job_id=job_id, topic=topic, mode=mode,
                                                  image_strategy="native_text", image_mode=config.image_mode,
                                                  speech_settings=speech_settings,
                                                  thumbnail_enabled=thumbnail_enabled is not False,
                                                  script_input_hash=script_input_hash, script_pauses=pauses, fps=config.fps))
        if script is not None:
            atomic_bytes(store.path("input_script.txt"), script_input.encode("utf-8"))
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
            if manifest.script_pauses:
                narration, pauses = parse_script_input(narration)
                if pauses != manifest.script_pauses:
                    raise PipelineError("Saved pause instructions do not match the input script")
            return self.save_script(make_script(narration))
        research = self.json_task("research", {"topic": manifest.topic, "text_generation": self.config.text_cache_settings, "prompt": RESEARCH_PROMPT},
                                  "sources.json", lambda: self.providers.research(manifest.topic))
        draft = self.json_task("script.draft", {"research": research, "topic": manifest.topic,
                              "text_generation": self.config.text_cache_settings, "prompt": SCRIPT_PROMPT},
                              "draft.json", lambda: self.providers.script(manifest.topic, research))
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

    def metadata(self, script: Script) -> dict:
        return self.json_task("metadata", {"narration": script.narration,
                              "text_generation": self.config.text_cache_settings,
                              "prompt": METADATA_PROMPT, "schema": METADATA_SCHEMA},
                              "video_metadata.json", lambda: self.providers.metadata(script.narration),
                              validate=validate_metadata)

    def thumbnail_plan(self, script: Script, title: str) -> dict:
        def plan():
            data = self.providers.thumbnail(script.narration, title)
            validate_thumbnail(data)
            headline = data["headline"].strip().upper()
            return {"headline": headline, "prompt": thumbnail_prompt(headline, data["prompt"], STYLE, THUMBNAIL_LAYOUT)}

        def validate(data):
            validate_thumbnail(data)
            suffix = thumbnail_prompt(data["headline"], "", STYLE, THUMBNAIL_LAYOUT)
            if (data["headline"] != data["headline"].strip().upper()
                    or not data["prompt"].endswith(suffix) or not data["prompt"][:-len(suffix)].strip()):
                raise TemporaryError("Invalid thumbnail plan: missing uppercase headline or fixed style/layout instructions")

        return self.json_task("thumbnail.plan", {"narration": script.narration, "title": title,
                              "text_generation": self.config.text_cache_settings, "schema": THUMBNAIL_SCHEMA,
                              "prompt": THUMBNAIL_PROMPT, "style": STYLE, "layout": THUMBNAIL_LAYOUT},
                              "thumbnail_prompt.json", plan, validate=validate)

    def thumbnail(self, script: Script, title: str):
        """Prepare an independent thumbnail without blocking the video on failure."""
        store = self.store
        report = {"status": "waiting_for_image", "prompt": "thumbnail_prompt.json",
                  "image": "thumbnail.png", "error": None}
        try:
            plan = self.thumbnail_plan(script, title)
            destination = store.path("thumbnail.png")
            if store.manifest.image_mode == "external":
                if destination.is_file():
                    validation = media.check_image(destination)
                    store.task("thumbnail.image", {"prompt": plan["prompt"], "source": "external",
                               "image": validation["sha256"], "version": 1}, lambda: {
                                   "files": ["thumbnail.png"], "source": "external", "validation": validation})
                    report["status"] = "ready"
            else:
                store.task("thumbnail.image", {"prompt": plan["prompt"], "source": "api",
                           "image_generation": self.config.image_cache_settings, "version": 1}, lambda: {
                               "files": ["thumbnail.png"], "source": "api",
                               "validation": self.generate_image(plan["prompt"], destination)})
                report["status"] = "ready"
        except Exception as exc:
            report.update(status="failed", error=str(exc))
        try:
            write_json(store.path("thumbnail_report.json"), report)
            store.log("thumbnail_status", **report)
        except OSError:
            if store.progress is not None:
                store.progress.console("Warning: could not save the thumbnail report; video workflow continues.")
        if store.progress is not None and report["status"] != "ready":
            store.progress.console(f"Thumbnail {report['status']}; video workflow continues."
                                   + (f" {report['error']}" if report["error"] else ""))
        return report

    def speech(self, script: Script):
        store, manifest = self.store, self.store.manifest
        # Repeat the invariant at the provider boundary, including resumed jobs.
        if not script.narration or script.narration != normalize(script.narration):
            raise PipelineError("Speech blocked: narration must be nonempty and normalized")
        if self.config.speech_provider == "voicestudio":
            from .speech import voicestudio_speech
            return voicestudio_speech(store, script)
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
            duration = media.audio_duration(self.config, temporary)
            alignment = speech.get("alignment")
            if not alignment:
                raise PipelineError("Speech provider returned no original character alignment; realignment is required")
            pause_data = {}
            if manifest.script_pauses:
                with wave.open(str(temporary), "rb") as decoded:
                    source_frames = decoded.getnframes()
                alignment, insertions = pause_timing(script, alignment, source_frames, manifest.script_pauses)
                media.insert_silence(temporary, insertions)
                duration = (source_frames + sum(p["frames"] for p in insertions)) / SPEECH_SAMPLE_RATE
                pause_data["pauses"] = insertions
            words = aligned_words(script, alignment, duration)
            temporary.replace(audio)
            sentences = [{"id": sentence.id, "text": sentence.text,
                          "start": min(w.start for w in words if w.sentence_id == sentence.id),
                          "end": max(w.end for w in words if w.sentence_id == sentence.id)} for sentence in script.sentences]
            write_json(store.path("alignment.json"), {"characters": alignment, "words": [w.model_dump() for w in words],
                       "sentences": sentences, "audio_duration": duration, "audio_sha256": file_hash(audio), **pause_data})
            return {"files": ["audio/narration.wav", "alignment.json"], "duration": duration}

        inputs = {"speech": digest(speech), "script": script.model_dump(), "decode_version": 1}
        if manifest.script_pauses:
            inputs["pauses"] = [p.model_dump() for p in manifest.script_pauses]
            inputs["pause_version"] = 1
        result = store.task("speech.decode", inputs, decode)
        manifest.audio_path = "audio/narration.wav"
        manifest.audio_duration = result["duration"]
        store.save()
        from .models import Word
        return [Word.model_validate(w) for w in read_json(store.path("alignment.json"))["words"]]

    def visual_context(self, narration):
        return self.json_task("visual.context", {"narration": narration,
                              "text_generation": self.config.text_cache_settings, "style": STYLE,
                              "prompt": VISUAL_CONTEXT_PROMPT, "schema": VISUAL_CONTEXT_SCHEMA},
                              "visual_context.json", lambda: self.providers.visual_context(narration),
                              validate=validate_visual_context)

    def cue_plan(self, words, narration):
        """Select exact word boundaries in bounded, independently resumable sections."""
        ends, first = [], 0
        pauses = {pause.word_index for pause in self.store.manifest.script_pauses}
        while first < len(words):
            end = min(first + 240, len(words))
            if end < len(words):
                sentence_ends = [index for index in range(first + 1, end + 1)
                                 if words[index - 1].sentence_id != words[index].sentence_id]
                if sentence_ends:
                    end = sentence_ends[-1]
            context = {"narration": narration, "word_start": first, "word_end": end,
                       "required_pause_boundaries": sorted(index for index in pauses if first < index < end),
                       "words": [{"index": index, **words[index].model_dump()}
                                 for index in range(max(0, first - 24), min(len(words), end + 24))]}
            planned = self.json_task(
                f"cue.plan.{first:06}",
                {"context": context, "text_generation": self.config.text_cache_settings,
                 "prompt": CUE_SPLIT_PROMPT, "contract": CUE_SPLIT_CONTRACT,
                 "schema": CUE_SPLIT_SCHEMA, "version": 1},
                f"cue_plans/words_{first:06}.json",
                lambda context=context: self.providers.cue_plan(context, CUE_SPLIT_PROMPT),
                validate=lambda data, first=first, end=end, context=context: validate_cue_plan(
                    data, first, end, context["required_pause_boundaries"]))
            ends.extend(planned["cue_end_word_indices"])
            first = end
        validate_cue_plan({"cue_end_word_indices": ends}, 0, len(words),
                          [index for index in pauses if 0 < index < len(words)])
        return ends

    def visuals(self, words, narration=None, cue_end_word_indices=None):
        store, manifest = self.store, self.store.manifest
        arguments = {} if cue_end_word_indices is None else {"cue_end_word_indices": cue_end_word_indices}
        cues = segment_words(words, self.config.fps, manifest.audio_duration,
                             (p.word_index for p in manifest.script_pauses), **arguments)
        if store.progress is not None:
            store.progress.set_total((len(cues) + VISUAL_BATCH_SIZE - 1) // VISUAL_BATCH_SIZE, "batches")
        context = self.visual_context(narration if narration is not None else " ".join(w.text for w in words))
        # Bounded batches prevent an entire long video from exceeding structured-output limits.
        records = []
        for offset in range(0, len(cues), VISUAL_BATCH_SIZE):
            batch = cues[offset:offset+VISUAL_BATCH_SIZE]
            previous_scenes = records[-3:]
            name = f"visuals.{offset//VISUAL_BATCH_SIZE:03}"
            inputs = {"cues": [c.model_dump() for c in batch], "context": context,
                      "previous_scenes": previous_scenes, "text_generation": self.config.text_cache_settings,
                      "style": STYLE, "prompt": VISUAL_PROMPT, "visual_response_version": VISUAL_RESPONSE_VERSION}
            planned = self.json_task(name, inputs,
                                     f"visuals/batch_{offset//VISUAL_BATCH_SIZE:03}.json",
                                     lambda batch=batch, previous_scenes=previous_scenes: self.providers.visuals(batch, previous_scenes, context),
                                     validate=lambda planned, batch=batch: validate_visual_plan(batch, planned, context))
            records.extend(planned["cues"])
        if [r["id"] for r in records] != [c.id for c in cues]:
            raise PipelineError("Visual plan must cover every cue ID exactly once in order")
        for cue, record in zip(cues, records):
            for field in ("scene_group", "character_ids", "scene_description", "motion"):
                setattr(cue, field, record[field])
            cue.scene_description = scene_prompt(cue.scene_description, cue.character_ids, context)
            Cue.model_validate(cue.model_dump())
        manifest.cues = cues
        manifest.total_frames = cues[-1].end_frame
        write_json(store.path("transcript.json"), [c.model_dump() for c in cues])
        atomic_bytes(store.path("subtitles.en.srt"), subtitles(cues, manifest.audio_duration).encode("utf-8"))
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

        def split_prompts():
            size, remainder = divmod(len(prompts), 3)
            files, offset = [], 0
            for index in range(3):
                end = offset + size + (index < remainder)
                relative = f"image_prompts_{index + 1}.json"
                write_json(store.path(relative), prompts[offset:end])
                files.append(relative)
                offset = end
            return {"files": files}

        store.task("image.prompts.split", {"prompts": prompts, "split_version": 1}, split_prompts)
        if manifest.image_mode == "external":
            self.external_images(prompts)
            return
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

    def external_images(self, prompts):
        """Validate user-supplied files and checkpoint them without calling providers."""
        store, manifest = self.store, self.store.manifest
        store.path("images").mkdir(parents=True, exist_ok=True)
        report = {"status": "ready", "image_mode": "external", "total_images": len(prompts),
                  "ready_images": 0, "missing_images": [], "invalid_images": [], "expected_images": []}
        for cue, record in zip(manifest.cues, prompts):
            relative = f"images/{cue.id}.png"
            report["expected_images"].append({"cue_id": cue.id, "path": relative})
            cue.image_path = None
            try:
                path = store.path(relative)
                if not path.is_file():
                    report["missing_images"].append(relative)
                    continue
                validation = media.check_image(path)
                inputs = {"prompt": record["prompt"], "source": "external", "image": validation["sha256"], "version": 1}
                previous = manifest.stages.get(f"image.{cue.id}")
                if (previous and previous.result.get("source") == "external"
                        and previous.artifacts.get(relative) == validation["sha256"]
                        and previous.input_hash != digest(inputs)):
                    # Keep the original adoption checkpoint so repeated resumes
                    # cannot associate unchanged bytes with a different prompt.
                    raise PipelineError("External image was adopted for a different prompt; "
                                        "replace it using the current image_prompts.json before resuming")
                store.task(f"image.{cue.id}", inputs, lambda relative=relative, validation=validation: {
                    "files": [relative], "validation": validation, "source": "external"})
                cue.image_path = relative
                report["ready_images"] += 1
            except (PipelineError, OSError) as exc:
                report["invalid_images"].append({"path": relative, "error": str(exc)})
        if report["missing_images"] or report["invalid_images"]:
            report["status"] = "waiting_for_images"
        write_json(store.path("external_images.json"), report)
        store.save()
        if report["status"] == "waiting_for_images":
            raise ImagesPending(report)

    def run(self):
        store, manifest = self.store, self.store.manifest
        manifest.status, manifest.error = "running", None
        store.save()
        phases = (["narration", "metadata", "speech", "visuals", "images"] if manifest.mode in {"production", "script"} else [])
        if phases and manifest.thumbnail_enabled:
            phases.insert(2, "thumbnail")
        progress = WorkflowProgress(store, phases + ["render", "finish", "validate"])
        store.progress = progress
        try:
            if manifest.mode in {"production", "script"}:
                with progress.phase("narration"):
                    script = self.script()
                with progress.phase("metadata"):
                    metadata = self.metadata(script)
                if manifest.thumbnail_enabled:
                    with progress.phase("thumbnail"):
                        self.thumbnail(script, metadata["title"])
                with progress.phase("speech"):
                    words = self.speech(script)
                with progress.phase("visuals"):
                    boundaries = self.cue_plan(words, script.narration)
                    self.visuals(words, script.narration, boundaries)
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
        except ImagesPending as pending:
            manifest.status, manifest.error = "waiting_for_images", None
            store.save()
            store.log("waiting_for_images", ready=pending.report["ready_images"], total=pending.report["total_images"])
            progress.finish("waiting_for_images")
            return pending.report
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
