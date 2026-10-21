from __future__ import annotations

import math
import re
import textwrap

from .models import Cue, Script, Sentence, Word
from .store import PipelineError


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def make_script(text: str, references=(), enforce_length=True) -> Script:
    text = normalize(text)
    if enforce_length and not 4500 <= len(text) <= 5000:
        raise PipelineError(f"Narration has {len(text)} characters; required 4500–5000 before speech generation")
    sentences = []
    for paragraph_index, paragraph in enumerate(re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text), 1):
        for match in re.finditer(r'.+?(?:[.!?]["\u201d\u2019\']*(?=\s|$)|$)', paragraph.group(), re.DOTALL):
            raw = match.group()
            if not raw.strip():
                continue
            start = paragraph.start() + match.start() + len(raw) - len(raw.lstrip())
            end = paragraph.start() + match.end() - (len(raw) - len(raw.rstrip()))
            sentences.append(Sentence(id=f"sentence_{len(sentences)+1:04}", paragraph_id=f"paragraph_{paragraph_index:03}",
                                      text=text[start:end], char_start=start, char_end=end))
    if not sentences or "".join(text.split()) != "".join("".join(s.text.split()) for s in sentences):
        raise PipelineError("Sentence segmentation lost narration text")
    return Script(narration=text, character_count=len(text), sentences=sentences, source_references=list(references))


def aligned_words(script: Script, alignment: dict, duration: float) -> list[Word]:
    characters = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends = alignment.get("character_end_times_seconds", [])
    if "".join(characters) != script.narration:
        raise PipelineError("Speech alignment does not match the exact normalized narration; realignment is required")
    if not (len(characters) == len(starts) == len(ends)):
        raise PipelineError("Character alignment arrays have different lengths")
    if any(not math.isfinite(s) or not math.isfinite(e) or s < 0 or e < s or e > duration + 0.05
           for s, e in zip(starts, ends)) or any(a > b for a, b in zip(starts, starts[1:])):
        raise PipelineError("Character alignment has invalid or unordered timestamps")
    words = []
    for match in re.finditer(r"\S+", script.narration):
        sentence = next((s for s in script.sentences if s.char_start <= match.start() < s.char_end), None)
        if sentence is None:
            raise PipelineError("Word is not covered by a sentence")
        words.append(Word(text=match.group(), start=starts[match.start()], end=min(duration, ends[match.end()-1]),
                          char_start=match.start(), char_end=match.end(), sentence_id=sentence.id))
    if not words or any(w.end <= w.start for w in words):
        raise PipelineError("Speech has empty or zero-duration words")
    return words


def segment_words(words: list[Word], fps: int, duration: float) -> list[Cue]:
    groups = []
    first = 0
    for index, word in enumerate(words):
        last = index == len(words) - 1
        sentence_end = last or words[index + 1].sentence_id != word.sentence_id
        count = index - first + 1
        if last or (count >= 4 and sentence_end) or count >= 14 or word.end - words[first].start >= 5:
            groups.append((first, index + 1))
            first = index + 1
    # Merge boundaries that round onto the same frame. All text stays present.
    merged = []
    for first, end in groups:
        start_frame = 0 if not merged else round(words[first].start * fps)
        if merged and (start_frame <= merged[-1][2] or start_frame >= math.ceil(duration * fps)):
            merged[-1] = (merged[-1][0], end, merged[-1][2])
        else:
            merged.append((first, end, start_frame))
    cues = [Cue(id=f"cue_{i+1:04}", text=" ".join(w.text for w in words[first:end]),
                start=words[first].start, end=words[end-1].end, word_start=first, word_end=end)
            for i, (first, end, _) in enumerate(merged)]
    assign_frames(cues, fps, duration)
    return cues


def assign_frames(cues: list[Cue], fps: int, duration: float):
    if not cues:
        raise PipelineError("Timeline has no cues")
    total = math.ceil(duration * fps)
    boundaries = [0] + [round(c.start * fps) for c in cues[1:]] + [total]
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise PipelineError("Cue boundaries produce zero or negative frame durations")
    for cue, start, end in zip(cues, boundaries, boundaries[1:]):
        cue.start_frame, cue.end_frame = start, end
    return total


def srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    total_seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, second = divmod(total_seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02}:{minute:02}:{second:02},{milliseconds:03}"


def subtitles(cues: list[Cue], duration: float) -> str:
    blocks = []
    for index, cue in enumerate(cues):
        end = min(cue.end, duration)
        if end <= cue.start:
            raise PipelineError(f"Invalid subtitle interval for {cue.id}")
        wrapped = "\n".join(textwrap.wrap(cue.text, width=56, break_long_words=False, break_on_hyphens=False))
        blocks.append(f"{index+1}\n{srt_time(cue.start)} --> {srt_time(end)}\n{wrapped}")
    return "\n\n".join(blocks) + "\n"
