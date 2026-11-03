"""Acoustic refinement for words that Whisper cannot assign a duration."""
from __future__ import annotations

import math
import re
import sys
import unicodedata
import warnings

from .store import PipelineError


MODEL_NAME = "WAV2VEC2_ASR_BASE_960H"
SAMPLE_RATE = 16000


def targets_for_words(words, labels):
    """Keep original word boundaries while encoding acoustic model targets."""
    dictionary = {char: index for index, char in enumerate(labels)}
    targets, ranges = [], []
    for word in words:
        start = len(targets)
        normalized = unicodedata.normalize("NFKD", word).upper().replace("\u2019", "'")
        for part in re.findall(r"[A-Z']+|\d+(?:[:.,]\d+)*|[^\W_]+", normalized):
            if all(char in dictionary for char in part):
                targets.extend(dictionary[char] for char in part)
            else:
                # An acoustic wildcard handles spoken numerals and unsupported
                # letters without changing the narration or guessing durations.
                targets.append(len(labels))
        if len(targets) == start:
            raise PipelineError("Acoustic alignment cannot encode a narration word")
        ranges.append((start, len(targets)))
        targets.append(dictionary["|"])
    return targets[:-1], ranges


def load_model():
    import torchaudio
    if not hasattr(torchaudio.functional, "forced_align"):
        raise PipelineError('Acoustic alignment requires Torch/torchaudio 2.8; install the project ".[voicestudio]" extra')
    bundle = getattr(torchaudio.pipelines, MODEL_NAME)
    return bundle.get_model().eval(), bundle.get_labels()


def align_window(model, labels, samples, words):
    import torch
    import torchaudio
    targets, ranges = targets_for_words(words, labels)
    waveform = torch.from_numpy(samples).float().unsqueeze(0)
    with torch.inference_mode():
        emission, _ = model(waveform)
        emission = emission.log_softmax(-1)
        # The wildcard follows acoustic evidence, including words such as 2:14.
        wildcard = emission[:, :, 2:].max(-1, keepdim=True).values
        emission = torch.cat((emission, wildcard), dim=-1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="torchaudio.functional._alignment.forced_align has been deprecated.*")
            path, scores = torchaudio.functional.forced_align(emission, torch.tensor([targets]), blank=0)
        spans = torchaudio.functional.merge_tokens(path[0], scores[0].exp())
    if len(spans) != len(targets) or any(span.token != target for span, target in zip(spans, targets)):
        raise PipelineError("Acoustic alignment did not cover every target")
    frames = emission.size(1)
    return [(spans[a].start * len(samples) / frames / SAMPLE_RATE,
             spans[b-1].end * len(samples) / frames / SAMPLE_RATE) for a, b in ranges]


def refine_instant_words(samples, narration: str, tokens: list[dict]) -> list[dict]:
    """Re-align bounded context windows; never interpolate missing word timing."""
    words = narration.split()
    duration = len(samples) / SAMPLE_RATE
    grouped, instant_words = [], set()
    partial, previous_end = "", 0.0
    for token in tokens:
        try:
            part = "".join(token["word"].split())
            if not part:
                continue
            start, end = float(token["start"]), float(token["end"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PipelineError("Acoustic refinement rejected invalid coarse tokens") from exc
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= duration and start >= previous_end):
            raise PipelineError("Acoustic refinement rejected invalid or unordered coarse timing")
        previous_end = end
        index = len(grouped)
        if index >= len(words) or not words[index].startswith(partial + part):
            raise PipelineError("Acoustic refinement requires exact original word coverage")
        if not partial:
            combined = dict(token, start=start, end=end)
        else:
            combined.update(word=partial + part, end=end)
        partial += part
        if start == end:
            instant_words.add(index)
        if partial == words[index]:
            grouped.append(combined)
            partial = ""
    if partial or len(grouped) != len(words) or not grouped:
        raise PipelineError("Acoustic refinement requires exact original word coverage")
    windows = []
    for index in sorted(instant_words):
        lo, hi = max(0, index-7), min(len(words), index+8)
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = hi
        else:
            windows.append([lo, hi])
    if not windows:
        return tokens
    # Refine whole original words even when only one constituent token has
    # zero duration. Grouping never supplies replacement acoustic timing.
    tokens = grouped
    print(f"Refining word timing on CPU in {len(windows)} acoustic windows; first use downloads wav2vec2 weights.",
          file=sys.stderr, flush=True)
    model, labels = load_model()
    repaired = [dict(token) for token in tokens]
    for number, (lo, hi) in enumerate(windows, 1):
        start = 0 if lo == 0 else max(tokens[lo-1]["end"], tokens[lo]["start"] - 0.2)
        end = duration if hi == len(words) else min(tokens[hi]["start"], tokens[hi-1]["end"] + 0.2)
        first, last = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
        if last <= first:
            raise PipelineError("Acoustic alignment has no recording interval for failed words")
        spans = align_window(model, labels, samples[first:last], words[lo:hi])
        if len(spans) != hi-lo:
            raise PipelineError("Acoustic alignment lost narration words")
        for index, (word_start, word_end) in enumerate(spans, lo):
            if not 0 <= word_start < word_end <= (last-first) / SAMPLE_RATE:
                raise PipelineError("Acoustic alignment returned invalid or zero-duration word timing")
            repaired[index].update(start=first/SAMPLE_RATE + word_start, end=first/SAMPLE_RATE + word_end)
        print(f"Acoustic alignment: {number}/{len(windows)} windows completed.", file=sys.stderr, flush=True)
    return repaired
