from __future__ import annotations

import math
import re
import textwrap

from .models import Cue, Script, ScriptPause, Sentence, Word
from .store import PipelineError, TemporaryError


SPEECH_SAMPLE_RATE = 48000


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_script_input(text: str) -> tuple[str, list[ScriptPause]]:
    """Separate explicit pauses from narration without changing other text."""
    text = normalize(text)
    pieces, pauses = [], []
    offset, word_index = 0, 0
    for marker in re.finditer(r"\[pause\b[^\]\r\n]*(?:\]|(?=\r|\n|$))", text, re.IGNORECASE):
        match = re.fullmatch(r"\[pause=(\d+(?:\.\d+)?)s\]", marker.group())
        seconds = float(match[1]) if match else 0
        if not math.isfinite(seconds) or not 0 < seconds <= 60 or round(seconds * SPEECH_SAMPLE_RATE) < 1:
            raise PipelineError("Invalid pause marker; use [pause=1s] with a positive duration of at most 60 seconds")
        if ((marker.start() and not text[marker.start()-1].isspace())
                or (marker.end() < len(text) and not text[marker.end()].isspace())):
            raise PipelineError("Pause markers must be separated from words and other markers by whitespace")
        piece = text[offset:marker.start()]
        pieces.append(piece)
        word_index += len(piece.split())
        pauses.append(ScriptPause(word_index=word_index, seconds=seconds))
        offset = marker.end()
    pieces.append(text[offset:])
    narration = normalize("".join(pieces))
    if not narration:
        raise PipelineError("Narration cannot be empty")
    return narration, pauses


def make_script(text: str, references=()) -> Script:
    text = normalize(text)
    if not text:
        raise PipelineError("Narration cannot be empty")
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
    if alignment.get("format") == "word_alignment_v1":
        if alignment.get("narration") != script.narration:
            raise PipelineError("Speech alignment does not match the exact normalized narration; realignment is required")
        try:
            words = [Word.model_validate(word) for word in alignment["words"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise PipelineError("Invalid word alignment records") from exc
        matches = list(re.finditer(r"\S+", script.narration))
        if not words or len(matches) != len(words):
            raise PipelineError("Word alignment does not cover every narration word")
        for match, word in zip(matches, words):
            sentence = next((s for s in script.sentences if s.char_start <= match.start() < s.char_end), None)
            if (word.text != match.group() or (word.char_start, word.char_end) != match.span()
                    or sentence is None or word.sentence_id != sentence.id):
                raise PipelineError("Word alignment text or offsets differ from original narration")
            if word.end <= word.start or word.end > duration + 0.00001:
                raise PipelineError("Word alignment has zero-duration or out-of-bounds timestamps")
        if any(a.end > b.start + 0.00001 for a, b in zip(words, words[1:])):
            raise PipelineError("Word alignment has unordered or overlapping timestamps")
        return words
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


def pause_timing(script: Script, alignment: dict, source_frames: int,
                 pauses: list[ScriptPause]) -> tuple[dict, list[dict]]:
    """Offset verified provider timestamps by the exact inserted sample counts."""
    duration = source_frames / SPEECH_SAMPLE_RATE
    words = aligned_words(script, alignment, duration)
    word_format = alignment.get("format") == "word_alignment_v1"
    starts = [] if word_format else list(alignment["character_start_times_seconds"])
    ends = [] if word_format else list(alignment["character_end_times_seconds"])
    shifted_words = [w.model_copy() for w in words]
    insertions, added_frames, previous_index = [], 0, 0
    for pause in pauses:
        index = pause.word_index
        if index < previous_index or index > len(words):
            raise PipelineError("Pause positions must follow narration word order")
        if index == 0:
            source_frame, char_offset = 0, 0
        elif index == len(words):
            source_frame, char_offset = source_frames, len(script.narration)
        else:
            if words[index-1].end > words[index].start:
                raise PipelineError("Cannot insert a pause between overlapping speech word timestamps")
            source_frame = round(words[index-1].end * SPEECH_SAMPLE_RATE)
            char_offset = words[index-1].char_end
        frames = round(pause.seconds * SPEECH_SAMPLE_RATE)
        if frames < 1 or not 0 <= source_frame <= source_frames:
            raise PipelineError("Pause duration or position is outside the decoded audio")
        start = (source_frame + added_frames) / SPEECH_SAMPLE_RATE
        insertions.append({"word_index": index, "source_frame": source_frame, "frames": frames,
                           "start": start, "end": (source_frame + added_frames + frames) / SPEECH_SAMPLE_RATE})
        shift = frames / SPEECH_SAMPLE_RATE
        for word in shifted_words[index:]:
            word.start += shift
            word.end += shift
        for char_index in range(char_offset, len(starts)):
            starts[char_index] += shift
            ends[char_index] += shift
        added_frames += frames
        previous_index = index
    shifted = ({**alignment, "words": [w.model_dump() for w in shifted_words]} if word_format else
               {**alignment, "character_start_times_seconds": starts, "character_end_times_seconds": ends})
    aligned_words(script, shifted, (source_frames + added_frames) / SPEECH_SAMPLE_RATE)
    return shifted, insertions


def validate_cue_boundaries(ends, word_start, word_end, pause_boundaries=()):
    """Reject boundaries that lose words, reorder them, or span explicit pauses."""
    if (not isinstance(ends, list) or not ends or any(type(end) is not int for end in ends)
            or ends[-1] != word_end
            or any(left >= right for left, right in zip([word_start] + ends, ends))
            or not set(pause_boundaries) <= set(ends)):
        raise TemporaryError("Invalid cue plan: boundaries must cover the word range once and preserve explicit pauses")


def segment_words(words: list[Word], fps: int, duration: float, pause_word_indices=(),
                  cue_end_word_indices=None) -> list[Cue]:
    if cue_end_word_indices is not None:
        validate_cue_boundaries(cue_end_word_indices, 0, len(words),
                                [index for index in pause_word_indices if 0 < index < len(words)])
        groups = list(zip([0] + cue_end_word_indices[:-1], cue_end_word_indices))
        return cues_from_groups(words, groups, fps, duration)
    groups = []
    first = 0
    pause_word_indices = set(pause_word_indices)
    for index, word in enumerate(words):
        last = index == len(words) - 1
        sentence_end = last or words[index + 1].sentence_id != word.sentence_id
        count = index - first + 1
        limit_reached = count >= 14 or word.end - words[first].start >= 5
        nearby_sentence_end = False
        if limit_reached and not sentence_end:
            # Keep a short ending with its sentence, within bounded extra time.
            for following in range(index + 1, min(index + 4, len(words))):
                if following in pause_word_indices or words[following].end - word.end > 1.5:
                    break
                if following == len(words) - 1 or words[following + 1].sentence_id != word.sentence_id:
                    nearby_sentence_end = True
                    break
        # A fragment of an already split sentence must not absorb the next idea.
        continuation = first > 0 and words[first - 1].sentence_id == word.sentence_id
        if (last or index + 1 in pause_word_indices or (sentence_end and (count >= 4 or continuation))
                or (limit_reached and not nearby_sentence_end)):
            groups.append((first, index + 1))
            first = index + 1
    return cues_from_groups(words, groups, fps, duration)


def cues_from_groups(words, groups, fps, duration):
    """Build timed cues while retaining the shared frame-collapse handling."""
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
