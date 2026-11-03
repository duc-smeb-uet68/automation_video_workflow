import math

import pytest

from video_automation import ctc_alignment as ctc
from video_automation.speech import map_alignment
from video_automation.store import PipelineError
from video_automation.timing import make_script


LABELS = tuple("-|ABCDEFGHIJKLMNOPQRSTUVWXYZ'")


def test_acoustic_targets_preserve_word_ranges_and_handle_numerals():
    targets, ranges = ctc.targets_for_words(["Café,", "2:14", "don't!"], LABELS)
    assert "".join(LABELS[i] if i < len(LABELS) else "*" for i in targets) == "CAFE|*|DON'T"
    assert ranges == [(0, 4), (5, 6), (7, 12)]
    with pytest.raises(PipelineError, match="cannot encode"):
        ctc.targets_for_words(["..."], LABELS)


def test_refinement_replaces_only_acoustically_realigned_windows(monkeypatch):
    tokens = [{"word": f"word{i}", "start": float(i), "end": i+0.5} for i in range(35)]
    tokens[18]["end"] = tokens[18]["start"]
    text = " ".join(t["word"] for t in tokens)
    calls = []
    monkeypatch.setattr(ctc, "load_model", lambda: ("model", LABELS))
    def align(model, labels, samples, words):
        assert model == "model" and labels == LABELS
        calls.append(words)
        return [(i+0.2, i+0.5) for i in range(len(words))]
    monkeypatch.setattr(ctc, "align_window", align)
    repaired = ctc.refine_instant_words([0] * (36*16000), text, tokens)
    assert calls == [[f"word{i}" for i in range(11, 26)]]
    assert tokens[18]["start"] == tokens[18]["end"]
    assert repaired[18]["start"] < repaired[18]["end"]
    assert repaired[:11] == tokens[:11] and repaired[26:] == tokens[26:]
    assert len(map_alignment(make_script(text), repaired, 36)["words"]) == 35


@pytest.mark.parametrize("corruption", ["text", "negative", "nan", "overlap"])
def test_refinement_rejects_bad_source_before_loading_model(monkeypatch, corruption):
    tokens = [{"word": "Hello", "start": 0.0, "end": 0.5}, {"word": "world.", "start": 1.0, "end": 1.0}]
    if corruption == "text":
        tokens[0]["word"] = "Changed"
    elif corruption == "negative":
        tokens[0]["start"] = -1
    elif corruption == "nan":
        tokens[0]["end"] = math.nan
    else:
        tokens[1]["start"] = 0.1
    monkeypatch.setattr(ctc, "load_model", lambda: pytest.fail("must not load acoustic weights"))
    with pytest.raises(PipelineError):
        ctc.refine_instant_words([0] * 32000, "Hello world.", tokens)


def test_refinement_still_rejects_zero_duration_results(monkeypatch):
    tokens = [{"word": "Hello.", "start": 1.0, "end": 1.0}]
    monkeypatch.setattr(ctc, "load_model", lambda: ("model", LABELS))
    monkeypatch.setattr(ctc, "align_window", lambda *args: [(1.0, 1.0)])
    with pytest.raises(PipelineError, match="zero-duration"):
        ctc.refine_instant_words([0] * 32000, "Hello.", tokens)


def test_valid_timing_never_loads_refinement_model(monkeypatch):
    tokens = [{"word": "Hello.", "start": 0.0, "end": 1.0}]
    monkeypatch.setattr(ctc, "load_model", lambda: pytest.fail("must not load acoustic weights"))
    assert ctc.refine_instant_words([0] * 32000, "Hello.", tokens) is tokens


@pytest.mark.parametrize("instant_index", [1, 3, 4])
def test_refinement_groups_split_tokens_and_preserves_original_offsets(monkeypatch, instant_index):
    text = "You're 38%!\nCafé."
    tokens = [{"word": word, "start": float(i), "end": i + 0.5}
              for i, word in enumerate([" You", "'re", " 38", "%!", " Café."])]
    tokens[instant_index]["end"] = tokens[instant_index]["start"]
    original = [dict(token) for token in tokens]
    calls = []
    monkeypatch.setattr(ctc, "load_model", lambda: ("model", LABELS))
    def align(model, labels, samples, words):
        calls.append(words)
        return [(0.1, 1.5), (2.1, 3.5), (4.1, 4.5)]
    monkeypatch.setattr(ctc, "align_window", align)
    repaired = ctc.refine_instant_words([0] * 80000, text, tokens)
    assert calls == [["You're", "38%!", "Café."]]
    words = map_alignment(make_script(text), repaired, 5)["words"]
    assert [(w["text"], w["start"], w["end"]) for w in words] == [
        ("You're", 0.1, 1.5), ("38%!", 2.1, 3.5), ("Café.", 4.1, 4.5)]
    assert all(text[w["char_start"]:w["char_end"]] == w["text"] for w in words)
    assert tokens == original


@pytest.mark.parametrize("corruption", ["crosses_word", "missing", "overlap", "negative", "nan"])
def test_split_token_refinement_rejects_invalid_source_before_model_load(monkeypatch, corruption):
    tokens = [{"word": "You", "start": 0., "end": .4},
              {"word": "'re", "start": .4, "end": .4},
              {"word": " ready.", "start": .6, "end": 1.}]
    if corruption == "crosses_word":
        tokens[1]["word"], tokens[2]["word"] = "'rer", "eady."
    elif corruption == "missing":
        tokens.pop()
    elif corruption == "overlap":
        tokens[1]["start"] = .3
    elif corruption == "negative":
        tokens[1]["start"] = -.1
    else:
        tokens[1]["end"] = math.nan
    monkeypatch.setattr(ctc, "load_model", lambda: pytest.fail("must not load acoustic weights"))
    with pytest.raises(PipelineError):
        ctc.refine_instant_words([0] * 32000, "You're ready.", tokens)


def test_refinement_preserves_split_words_outside_acoustic_windows(monkeypatch):
    tokens = [{"word": f"word{i}", "start": float(i), "end": i + .5} for i in range(35)]
    text = " ".join(token["word"] for token in tokens)
    expected = map_alignment(make_script(text), tokens, 36)["words"]
    tokens[18]["end"] = tokens[18]["start"]
    tokens[:1] = [{"word": "word", "start": 0., "end": .25},
                  {"word": "0", "start": .25, "end": .5}]
    monkeypatch.setattr(ctc, "load_model", lambda: ("model", LABELS))
    monkeypatch.setattr(ctc, "align_window", lambda model, labels, samples, words:
                        [(i + .2, i + .5) for i in range(len(words))])
    repaired = ctc.refine_instant_words([0] * (36 * 16000), text, tokens)
    actual = map_alignment(make_script(text), repaired, 36)["words"]
    assert actual[:11] == expected[:11] and actual[26:] == expected[26:]


def test_acoustic_frame_conversion_keeps_last_word_inside_recording():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    np = pytest.importorskip("numpy")
    class Model:
        def __call__(self, waveform):
            emission = torch.full((1, 187, 4), -10.0)
            emission[0, :-1, 2] = 10.0
            emission[0, -1, 3] = 10.0
            return emission, None
    samples = np.zeros(60160, dtype=np.float32)
    bounds = ctc.align_window(Model(), ("-", "|", "A", "B"), samples, ["AB"])
    assert bounds == [(0.0, len(samples) / 16000)]
