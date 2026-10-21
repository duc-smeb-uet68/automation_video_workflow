from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .models import Cue
from .store import PipelineError, atomic_bytes, file_hash, write_json
from .timing import assign_frames, normalize, subtitles


def run(command: list[str]) -> str:
    result = subprocess.run([str(v) for v in command], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise PipelineError(f"{Path(command[0]).name} failed ({result.returncode}): {result.stderr[-2500:]}")
    return result.stdout + result.stderr


def ffmpeg(config, *args) -> str:
    return run([config.ffmpeg, "-hide_banner", "-nostdin", "-y", *map(str, args)])


def probe(config, path: Path) -> dict:
    return json.loads(run([config.ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]))


def audio_duration(config, path: Path) -> float:
    info = probe(config, path)
    streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if not streams:
        raise PipelineError(f"No audio stream: {path}")
    duration = float(streams[0].get("duration") or info["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise PipelineError(f"Invalid audio duration: {path}")
    return duration


def check_image(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width < 320 or height < 180 or abs(width / height - 16 / 9) > 0.06:
                raise PipelineError(f"Image must be approximately 16:9 and at least 320x180: {path}")
            return {"width": width, "height": height, "sha256": file_hash(path)}
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Invalid image {path.name}: {exc}") from exc


def overlay(config, text: str, output: Path) -> dict:
    width, height = config.width, config.height
    font_paths = [config.font_path, Path("C:/Windows/Fonts/arialbd.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
    font_path = next((p for p in font_paths if p and p.is_file()), None)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    margin = max(16, width // 24)
    allowed_width = width - margin * 4
    for size in range(max(16, height // 20), 11, -1):
        font = ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)
        lines = []
        for word in text.split():
            candidate = f"{lines[-1]} {word}" if lines else word
            if draw.textlength(candidate, font=font) <= allowed_width:
                if lines:
                    lines[-1] = candidate
                else:
                    lines.append(word)
            else:
                lines.append(word)
        wrapped = "\n".join(lines)
        box = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6)
        if box[2] - box[0] <= allowed_width and box[3] - box[1] <= height * 0.25:
            break
    else:
        raise PipelineError("Overlay text does not fit in the configured frame")
    if not wrapped:
        raise PipelineError("Overlay text is empty")
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    left = (width - box_width) // 2
    top = margin * 2
    draw.rounded_rectangle((left-margin, top-margin, left+box_width+margin, top+box_height+margin),
                           radius=12, fill=(255, 250, 235, 245))
    draw.multiline_text((left-box[0], top-box[1]), wrapped, font=font, fill=(160, 25, 20, 255), spacing=6, align="center")
    bounds = canvas.getbbox()
    if not bounds or not (0 <= bounds[0] < bounds[2] <= width and 0 <= bounds[1] < bounds[3] <= height):
        raise PipelineError("Overlay exceeds frame bounds")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {"text": text, "bounds": list(bounds)}


def prepare_image(config, source: Path, destination: Path, overlay_path: Path | None):
    with Image.open(source) as image:
        image = ImageOps.pad(image.convert("RGB"), (config.width, config.height), color="#fffaf0")
        if overlay_path:
            with Image.open(overlay_path) as labels:
                image = Image.alpha_composite(image.convert("RGBA"), labels.convert("RGBA")).convert("RGB")
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)


def render(store):
    config, manifest = store.config, store.manifest
    manifest.total_frames = assign_frames(manifest.cues, config.fps, manifest.audio_duration)
    manifest.fps = config.fps
    store.save()
    clips = []
    for cue in manifest.cues:
        image = store.path(cue.image_path)
        check_image(image)
        clip_relative = f"render/clips/{cue.id}.mp4"
        overlay_path = store.path(cue.overlay_path) if cue.overlay_path else None
        inputs = {"image": file_hash(image), "overlay": file_hash(overlay_path) if overlay_path else None,
                  "frames": cue.end_frame-cue.start_frame, "motion": cue.motion,
                  "settings": config.model_dump(include={"width", "height", "fps", "crf", "encoder_preset"})}

        def encode(cue=cue, image=image, overlay_path=overlay_path, relative=clip_relative):
            prepared = store.path(f"render/prepared/{cue.id}.png")
            prepare_image(config, image, prepared, overlay_path)
            frames = cue.end_frame - cue.start_frame
            # Reuse the original builder's restrained 1.5% centered push. Labels remain stationary.
            if cue.motion == "slow_push" and not cue.overlay_text:
                filters = (f"zoompan=z='1+0.015*on/{max(1, frames-1)}':x='iw/2-iw/zoom/2':"
                           f"y='ih/2-ih/zoom/2':d={frames}:s={config.width}x{config.height}:fps={config.fps}")
            else:
                filters = f"fps={config.fps}"
            output = store.path(relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.mp4")
            ffmpeg(config, "-loglevel", "error", "-loop", "1", "-framerate", config.fps, "-i", prepared,
                   "-vf", filters, "-frames:v", frames, "-an", "-c:v", "libx264", "-preset", config.encoder_preset,
                   "-crf", config.crf, "-threads", config.threads, "-pix_fmt", "yuv420p", temporary)
            temporary.replace(output)
            return {"files": [relative], "frames": frames}

        store.task(f"render.{cue.id}", inputs, encode)
        clips.append(clip_relative)

    inputs = {"clips": [file_hash(store.path(p)) for p in clips],
              "audio": file_hash(store.path(manifest.audio_path)), "frames": manifest.total_frames}

    def assemble():
        # Relative generated filenames avoid platform-specific concat escaping.
        concat = store.path("render/clips.txt")
        atomic_bytes(concat, "".join(f"file 'clips/{Path(p).name}'\n" for p in clips).encode("utf-8"))
        output = store.path("rough.mp4")
        temporary = output.with_suffix(".tmp.mp4")
        ffmpeg(config, "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat,
               "-i", store.path(manifest.audio_path), "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", manifest.total_frames/config.fps,
               "-movflags", "+faststart", temporary)
        temporary.replace(output)
        metadata = {"fps": config.fps, "total_frames": manifest.total_frames,
                    "audio_duration": manifest.audio_duration,
                    "cues": [{"id": c.id, "start_frame": c.start_frame, "end_frame": c.end_frame} for c in manifest.cues]}
        write_json(store.path("render_metadata.json"), metadata)
        return {"files": ["rough.mp4", "render_metadata.json"]}

    store.task("rough", inputs, assemble)


def loudness(config, audio: Path) -> dict:
    log = ffmpeg(config, "-i", audio, "-vn", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-")
    matches = re.findall(r'\{\s*"input_i".*?\}', log, re.DOTALL)
    if not matches:
        raise PipelineError("FFmpeg did not return loudness measurements")
    result = json.loads(matches[-1])
    if not all(math.isfinite(float(result[k])) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
        raise PipelineError("Narration has no measurable speech audio")
    return result


def finish(store):
    config, manifest = store.config, store.manifest
    music = config.music_path
    if music and not music.is_file():
        raise PipelineError(f"Configured music file does not exist: {music}")
    inputs = {"rough": file_hash(store.path("rough.mp4")), "audio": file_hash(store.path(manifest.audio_path)),
              "music": file_hash(music) if music else None, "cues": [c.model_dump() for c in manifest.cues],
              "fps": config.fps, "total_frames": manifest.total_frames, "finishing_version": 1}

    def action():
        measurements = loudness(config, store.path(manifest.audio_path))
        normalizer = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={measurements['input_i']}:"
                      f"measured_TP={measurements['input_tp']}:measured_LRA={measurements['input_lra']}:"
                      f"measured_thresh={measurements['input_thresh']}:offset={measurements['target_offset']}:linear=true")
        duration = manifest.total_frames / config.fps
        voice = f"[1:a]{normalizer},aresample=48000,apad,atrim=duration={duration}[voice]"
        command = ["-loglevel", "error", "-i", store.path("rough.mp4"), "-i", store.path(manifest.audio_path)]
        if music:
            command += ["-stream_loop", "-1", "-i", music]
            fade = min(1.5, duration/4)
            filters = (voice + f";[voice]asplit=2[speech][sidechain];[2:a]aresample=48000,"
                       f"atrim=duration={duration},asetpts=PTS-STARTPTS,volume=0.07,"
                       f"afade=t=in:d={fade},afade=t=out:st={duration-fade}:d={fade}[music];"
                       "[music][sidechain]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=250[ducked];"
                       "[speech][ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.89:level=0:latency=1[mix]")
        else:
            filters = voice + ";[voice]alimiter=limit=0.89:level=0:latency=1[mix]"
        output = store.path("final.mp4")
        temporary = output.with_suffix(".tmp.mp4")
        ffmpeg(config, *command, "-filter_complex", filters, "-map", "0:v:0", "-map", "[mix]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", duration,
               "-movflags", "+faststart", temporary)
        temporary.replace(output)
        atomic_bytes(store.path("subtitles.en.srt"), subtitles(manifest.cues, manifest.audio_duration).encode("utf-8"))
        write_json(store.path("audio_measurements.json"), {"narration": measurements, "music": str(music) if music else None})
        return {"files": ["final.mp4", "subtitles.en.srt", "audio_measurements.json"]}

    store.task("finish", inputs, action)


def audio_health(config, path: Path, duration: float) -> dict:
    log = ffmpeg(config, "-i", path, "-vn", "-af",
                 f"astats=metadata=0:reset=0,silencedetect=noise=-45dB:d={config.max_silence_seconds}", "-f", "null", "-")
    peaks = [float(v) for v in re.findall(r"Peak level dB: ([-+\w.]+)", log)]
    starts = [float(v) for v in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(v) for v in re.findall(r"silence_end: ([\d.]+)", log)]
    intervals = [{"start": start, "end": ends[i] if i < len(ends) else duration} for i, start in enumerate(starts)]
    return {"peak_dbfs": max(peaks) if peaks else None, "long_silences": intervals}


def validate(store) -> dict:
    config, manifest = store.config, store.manifest
    checks, details = {}, {}

    def check(name, action):
        try:
            value = action()
            checks[name] = bool(value)
            details[name] = value
        except Exception as exc:
            checks[name] = False
            details[name] = str(exc)

    try:
        script_text = store.path("script.txt").read_text(encoding="utf-8")
        script_present = bool(script_text)
    except OSError as exc:
        script_text, script_present = "", False
        details["script_error"] = str(exc)
    generated_speech = manifest.mode in {"production", "script"}
    checks["narration_length"] = script_present and (not generated_speech or (
        4500 <= len(normalize(script_text)) <= 5000 and script_text == normalize(script_text)
        and len(script_text) == manifest.script_character_count))
    details["narration_length"] = {"characters": len(normalize(script_text)), "enforced": generated_speech}
    if manifest.mode == "script":
        details["source_audit"] = "Skipped: narration supplied by the user."

    def images():
        for cue in manifest.cues:
            check_image(store.path(cue.image_path))
        return bool(manifest.cues)

    check("cue_images", images)
    check("timeline_coverage", lambda: bool(manifest.cues) and manifest.cues[0].start_frame == 0
          and manifest.cues[-1].end_frame == manifest.total_frames
          and all(c.end_frame > c.start_frame for c in manifest.cues)
          and all(a.end_frame == b.start_frame for a, b in zip(manifest.cues, manifest.cues[1:])))
    words_path = store.path("alignment.json")
    if words_path.is_file():
        try:
            alignment = json.loads(words_path.read_text(encoding="utf-8"))
            words = alignment["words"]
            expected = " ".join(w["text"] for w in words)
            check("alignment_audio_match", lambda: alignment["audio_sha256"] == file_hash(store.path(manifest.audio_path))
                  and abs(alignment["audio_duration"]-manifest.audio_duration) < 0.00001)
            check("spoken_text_coverage", lambda: " ".join(c.text for c in manifest.cues) == expected
                  == " ".join(script_text.split()) and manifest.cues[0].word_start == 0
                  and manifest.cues[-1].word_end == len(words)
                  and all(a.word_end == b.word_start for a, b in zip(manifest.cues, manifest.cues[1:])))
            check("cue_word_timing", lambda: all(
                abs(c.start-words[c.word_start]["start"]) < 0.00001
                and abs(c.end-words[c.word_end-1]["end"]) < 0.00001
                and c.text == " ".join(w["text"] for w in words[c.word_start:c.word_end]) for c in manifest.cues))
        except Exception as exc:
            checks["spoken_text_coverage"] = checks["alignment_audio_match"] = False
            details["alignment_error"] = str(exc)
    else:
        check("spoken_text_coverage", lambda: " ".join(c.text for c in manifest.cues) == " ".join(script_text.split()))
        if generated_speech:
            checks["alignment_audio_match"] = False
            details["alignment_error"] = "Generated speech requires alignment.json"
    check("caption_bounds", lambda: all(0 <= c.start < c.end <= manifest.audio_duration + 0.001 for c in manifest.cues)
          and store.path("subtitles.en.srt").read_text(encoding="utf-8") == subtitles(manifest.cues, manifest.audio_duration))

    def overlays():
        for cue in manifest.cues:
            if cue.overlay_text:
                with Image.open(store.path(cue.overlay_path)) as image:
                    bounds = image.getbbox()
                    if image.size != (config.width, config.height) or not bounds:
                        return False
                    if not (0 <= bounds[0] < bounds[2] <= config.width and 0 <= bounds[1] < bounds[3] <= config.height):
                        return False
                if cue.motion != "hold":
                    return False
        return True

    check("overlay_bounds", overlays)
    try:
        info = probe(config, store.path("final.mp4"))
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        video_seconds, audio_seconds = float(video["duration"]), float(audio["duration"])
        details["durations"] = {"video": video_seconds, "audio": audio_seconds, "narration": manifest.audio_duration}
        checks["duration_match"] = abs(video_seconds-audio_seconds) <= 1/config.fps + 0.00001 and abs(video_seconds-manifest.audio_duration) <= 1/config.fps + 0.00001
        checks["output_format"] = video["width"] == config.width and video["height"] == config.height and video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p" and video["r_frame_rate"] == f"{config.fps}/1" and int(video["nb_frames"]) == manifest.total_frames
    except Exception as exc:
        checks["duration_match"] = checks["output_format"] = False
        details["probe_error"] = str(exc)

    def decode():
        ffmpeg(config, "-v", "error", "-xerror", "-i", store.path("final.mp4"), "-f", "null", "-")
        return True

    check("decode", decode)
    try:
        health = audio_health(config, store.path("final.mp4"), manifest.total_frames/config.fps)
        details["audio_health"] = health
        checks["no_clipping"] = health["peak_dbfs"] is not None and health["peak_dbfs"] < -0.1
        checks["no_long_silence"] = not health["long_silences"]
        source_health = audio_health(config, store.path(manifest.audio_path), manifest.audio_duration)
        details["narration_health"] = source_health
        checks["narration_no_clipping"] = source_health["peak_dbfs"] is not None and source_health["peak_dbfs"] < -0.1
        checks["narration_no_long_silence"] = not source_health["long_silences"]
    except Exception as exc:
        checks["no_clipping"] = checks["no_long_silence"] = False
        details["audio_error"] = str(exc)
    report = {"passed": all(checks.values()), "mode": manifest.mode, "checks": checks, "details": details}
    write_json(store.path("quality_report.json"), report)
    store.log("validation", passed=report["passed"], checks=checks)
    if not report["passed"]:
        raise PipelineError("Final validation failed: " + ", ".join(k for k, v in checks.items() if not v))
    return report
