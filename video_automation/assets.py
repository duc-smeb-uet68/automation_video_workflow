from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from PIL import Image, ImageDraw

from . import media
from .models import Cue, Manifest
from .pipeline import Pipeline
from .store import PipelineError, Store, atomic_bytes, read_json, write_json
from .timing import make_script


def parse_srt(path: Path) -> list[dict]:
    def seconds(stamp):
        h, m, s = stamp.replace(",", ".").split(":")
        return int(h)*3600 + int(m)*60 + float(s)
    result = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()):
        lines = block.splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            raise PipelineError("Malformed SRT block")
        start, end = lines[1].split(" --> ")
        result.append({"start": seconds(start), "end": seconds(end), "text": " ".join(lines[2:])})
    if not result:
        raise PipelineError("Subtitle file has no cues")
    return result


def empty_store(config, job_id, topic, mode):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job_id):
        raise PipelineError("Job ID must contain only letters, digits, underscores and hyphens")
    folder = config.jobs_dir / job_id
    if (folder / "manifest.json").exists():
        raise PipelineError(f"Job already exists; use resume: {folder}")
    return Store(folder, config, Manifest(job_id=job_id, topic=topic, mode=mode, fps=config.fps))


def import_assets(config, manifest_path, audio_path, subtitles_path, job_id, limit_cues=None):
    legacy = read_json(manifest_path)
    clips = legacy["prompts"]
    captions = parse_srt(subtitles_path)
    corrections = legacy.get("generation_notes", {}).get("corrections", [])
    if any(not correction.get("applied") for correction in corrections):
        raise PipelineError("Legacy manifest contains unapplied image corrections")
    if len(clips) != len(captions):
        raise PipelineError("Legacy images and subtitle cues do not have matching counts")
    if limit_cues is not None and not 1 <= limit_cues <= len(clips):
        raise PipelineError("Fixture cue limit is outside the available cue range")
    count = limit_cues or len(clips)
    duration = media.audio_duration(config, audio_path)
    cutoff = captions[count]["start"] if count < len(captions) else duration
    store = empty_store(config, job_id, "Imported existing educational video assets", "fixture" if limit_cues else "imported")
    output = store.path("audio/narration.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    media.ffmpeg(config, "-loglevel", "error", "-i", audio_path, "-t", cutoff, "-vn", "-ac", "1", "-ar", "48000", output)
    store.manifest.audio_duration = media.audio_duration(config, output)
    store.manifest.audio_path = "audio/narration.wav"
    # Retain the existing builder's approved motion decisions when available.
    decisions_path = subtitles_path.parent / "edit_decisions.csv"
    decisions = {}
    if decisions_path.is_file():
        with decisions_path.open(encoding="utf-8-sig", newline="") as stream:
            decisions = {row["filename"]: row.get("motion", "static") for row in csv.DictReader(stream)}
    word_cursor = 0
    for index, (clip, caption) in enumerate(zip(clips[:count], captions[:count])):
        if clip.get("status") != "completed":
            raise PipelineError(f"Legacy image is not approved: {clip['filename']}")
        fields = [int(v) for v in clip["timestamp"].split(":")]
        timestamp = sum(value * 60**power for power, value in enumerate(reversed(fields)))
        if abs(timestamp-caption["start"]) > 0.034:
            raise PipelineError(f"Legacy timestamp differs from subtitle timestamp: {clip['filename']}")
        source = (manifest_path.parent / clip["filename"]).resolve()
        if not source.is_relative_to(manifest_path.parent.resolve()):
            raise PipelineError("Legacy image path escapes its source directory")
        media.check_image(source)
        relative = f"images/cue_{index+1:04}.png"
        atomic_bytes(store.path(relative), source.read_bytes())
        word_count = len(caption["text"].split())
        end = min(caption["end"], store.manifest.audio_duration)
        if index + 1 < count and end > captions[index+1]["start"]:
            raise PipelineError("Legacy subtitles overlap")
        store.manifest.cues.append(Cue(id=f"cue_{index+1:04}", text=caption["text"], start=caption["start"],
                                     end=end, word_start=word_cursor, word_end=word_cursor+word_count,
                                     image_path=relative, scene_group="legacy", scene_description=clip.get("prompt", ""),
                                     motion="slow_push" if "slow" in decisions.get(clip["filename"], "") else "hold"))
        word_cursor += word_count
    script = make_script("\n\n".join(c.text for c in store.manifest.cues))
    atomic_bytes(store.path("script.txt"), script.narration.encode("utf-8"))
    write_json(store.path("script.json"), script.model_dump())
    write_json(store.path("import_provenance.json"), {"manifest": str(manifest_path.resolve()), "audio": str(audio_path.resolve()),
               "subtitles": str(subtitles_path.resolve()), "cue_count": count, "alignment": "supplied cue timings; no word alignment claimed"})
    store.manifest.script_character_count = script.character_count
    store.save()
    return Pipeline(store)


def fixture(config, job_id="renderer_fixture"):
    """Portable renderer smoke test with synthetic audio; this is not a generated episode."""
    store = empty_store(config, job_id, "Synthetic renderer verification fixture", "fixture")
    audio = store.path("audio/narration.wav")
    audio.parent.mkdir(parents=True, exist_ok=True)
    media.ffmpeg(config, "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=3.117", "-c:a", "pcm_s16le", audio)
    store.manifest.audio_path = "audio/narration.wav"
    store.manifest.audio_duration = media.audio_duration(config, audio)
    texts = ["Renderer fixture: stationary scene.", "Gentle camera movement remains restrained.", "Source image label: 38%"]
    for index, (text, color) in enumerate(zip(texts, ["#f2dfb3", "#c6d9b4", "#e9d0ad"])):
        relative = f"images/cue_{index+1:04}.png"
        image = Image.new("RGB", (640, 360), color)
        draw = ImageDraw.Draw(image)
        draw.ellipse((260, 90, 380, 210), fill="white", outline="black", width=5)
        draw.line((320, 210, 320, 320), fill="black", width=5)
        draw.ellipse((290, 135, 300, 145), fill="black")
        draw.ellipse((340, 135, 350, 145), fill="black")
        if index == 2:
            # The fixture simulates text already drawn inside a provider's source image.
            draw.text((40, 30), "38%", fill=(160, 25, 20), font_size=32)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        atomic_bytes(store.path(relative), buffer.getvalue())
        cue = Cue(id=f"cue_{index+1:04}", text=text, start=float(index), end=min(index+0.9, store.manifest.audio_duration),
                  word_start=index*5, word_end=(index+1)*5, image_path=relative,
                  motion="slow_push" if index == 1 else "hold")
        store.manifest.cues.append(cue)
    script = make_script("\n\n".join(texts))
    atomic_bytes(store.path("script.txt"), script.narration.encode("utf-8"))
    write_json(store.path("script.json"), script.model_dump())
    store.manifest.script_character_count = script.character_count
    store.save()
    return Pipeline(store)
