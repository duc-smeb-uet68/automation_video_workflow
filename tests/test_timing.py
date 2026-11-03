import math
import re

import pytest

from video_automation.models import Word
from video_automation.timing import make_script, segment_words, subtitles


def timed_words(text, seconds=0.3):
    script = make_script(text)
    return [Word(text=match.group(), start=index * seconds, end=(index + 1) * seconds,
                 char_start=match.start(), char_end=match.end(),
                 sentence_id=next(sentence.id for sentence in script.sentences
                                  if sentence.char_start <= match.start() < sentence.char_end))
            for index, match in enumerate(re.finditer(r"\S+", text))]


def test_octopus_sentence_ending_stays_with_its_scene():
    first = "Their blood is blue because it uses a copper-containing protein called hemocyanin to carry oxygen."
    second = "Even stranger, the main heart pauses when an octopus swims."
    words = timed_words(first + " " + second)
    cues = segment_words(words, 30, words[-1].end)

    assert [cue.text for cue in cues] == [first, second]
    assert cues[1].start == words[15].start
    assert "carry oxygen." in subtitles(cues, words[-1].end)
    assert cues[0].end_frame == cues[1].start_frame == round(words[15].start * 30)


@pytest.mark.parametrize("tail", [1, 2, 3])
def test_word_limit_keeps_nearby_sentence_ending(tail):
    first = " ".join(["word"] * (14 + tail)) + "."
    second = "A new idea starts here."
    words = timed_words(first + " " + second, seconds=0.25)

    assert [cue.text for cue in segment_words(words, 30, words[-1].end)] == [first, second]


def test_duration_limit_keeps_nearby_sentence_ending():
    first = "We follow the blood through each heart and around the body."
    second = "A new idea starts here."
    words = timed_words(first + " " + second, seconds=0.5)

    assert [cue.text for cue in segment_words(words, 30, words[-1].end)] == [first, second]


def test_slow_sentence_tail_does_not_merge_with_next_idea():
    first = "Their blood is blue because it uses a copper-containing protein called hemocyanin to carry oxygen."
    second = "Even stranger, the main heart pauses when an octopus swims."
    words = timed_words(first + " " + second, seconds=0.25)
    words[14].end += 2
    for word in words[15:]:
        word.start += 2
        word.end += 2
    cues = segment_words(words, 30, words[-1].end)

    assert [cue.text for cue in cues] == [first.rsplit(" ", 1)[0], "oxygen.", second]
    assert cues[0].word_end == 14
    assert cues[2].start == words[15].start


def test_sentence_lookahead_preserves_explicit_pause():
    words = timed_words(" ".join(["word"] * 16) + ". A new idea starts here.", seconds=0.25)
    for word in words[14:]:
        word.start += 1
        word.end += 1
    cues = segment_words(words, 30, words[-1].end, pause_word_indices=[14])

    assert [(cue.word_start, cue.word_end) for cue in cues] == [(0, 14), (14, 16), (16, 21)]
    assert cues[0].end == words[13].end
    assert cues[1].start == words[14].start
    assert cues[0].end_frame == cues[1].start_frame


def test_long_sentence_stays_bounded_and_preserves_all_words():
    words = timed_words(" ".join(["word"] * 40) + ".", seconds=0.25)
    duration = words[-1].end + 0.02
    cues = segment_words(words, 30, duration)

    assert [cue.word_end - cue.word_start for cue in cues] == [14, 14, 12]
    assert " ".join(cue.text for cue in cues) == " ".join(word.text for word in words)
    assert cues[0].start_frame == 0
    assert cues[-1].end_frame == math.ceil(duration * 30)
    assert all(left.end_frame == right.start_frame for left, right in zip(cues, cues[1:]))


def test_short_complete_sentences_keep_existing_grouping():
    words = timed_words("One. Two. Three. Four. Five. Six. Seven. Eight.")
    cues = segment_words(words, 30, words[-1].end)

    assert [cue.text for cue in cues] == ["One. Two. Three. Four.", "Five. Six. Seven. Eight."]
