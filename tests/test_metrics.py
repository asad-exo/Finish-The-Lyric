import pytest

from finish_the_lyric import metrics

REFERENCES = [["the night is young", "and so are we"], ["hold on to the feeling"]]


def test_identical_output_scores_full_marks():
    assert metrics.bleu(REFERENCES, REFERENCES) == pytest.approx(100)
    assert metrics.chrf(REFERENCES, REFERENCES) == pytest.approx(100)


def test_bleu_works_on_words_not_characters():
    # Same letters in a different order share many characters but no word sequences.
    shuffled = [["gnuoy si thgin eht", "ew era os dna"], ["gnileef eht ot no dloh"]]
    assert metrics.bleu(shuffled, REFERENCES) < 1


def test_distinct_n():
    assert metrics.distinct_n([["la la la la"]], 2) == pytest.approx(1 / 3)
    assert metrics.distinct_n([["one two three"]], 2) == 1.0
    assert metrics.distinct_n([[]], 2) == 0.0


def test_copy_rate_ignores_case_and_punctuation():
    outputs = [["Hey, hey!", "something new"]]
    prompts = [["hey hey", "an earlier line"]]
    assert metrics.copy_rate(outputs, prompts) == 0.5


def test_words_per_line():
    assert metrics.words_per_line([["one two", "three four five six"]]) == 3.0
