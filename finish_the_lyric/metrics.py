"""Scores for generated song endings.

Each output and reference is a list of lines. BLEU and chrF compare an output with the
song's real ending; the other measures describe the output on its own.
"""

import sacrebleu

from .text import normalise, words


def _joined(lines: list[str]) -> str:
    return " ".join(lines)


def bleu(outputs: list[list[str]], references: list[list[str]]) -> float:
    """Corpus BLEU (0–100) on words, ignoring case. Rewards exact word sequences."""
    return sacrebleu.corpus_bleu(
        [_joined(o) for o in outputs], [[_joined(r) for r in references]], lowercase=True
    ).score


def chrf(outputs: list[list[str]], references: list[list[str]]) -> float:
    """Corpus chrF (0–100): character n-gram overlap, which gives partial credit for close words."""
    return sacrebleu.corpus_chrf([_joined(o) for o in outputs], [[_joined(r) for r in references]]).score


def distinct_n(outputs: list[list[str]], n: int = 2) -> float:
    """Share of word n-grams across all outputs that are unique. Low values mean repetitive text."""
    grams = []
    for lines in outputs:
        w = words(_joined(lines))
        grams.extend(zip(*(w[i:] for i in range(n))))
    return len(set(grams)) / len(grams) if grams else 0.0


def copy_rate(outputs: list[list[str]], prompts: list[list[str]]) -> float:
    """Share of output lines that repeat a line from earlier in the same song.

    Songs repeat themselves (a chorus comes back), so some copying is expected. Comparing
    with the real endings shows whether a model copies too much or too little.
    """
    copied = total = 0
    for lines, prompt in zip(outputs, prompts):
        earlier = {normalise(line) for line in prompt}
        for line in lines:
            total += 1
            copied += normalise(line) in earlier
    return copied / total if total else 0.0


def words_per_line(outputs: list[list[str]]) -> float:
    counts = [len(words(line)) for lines in outputs for line in lines]
    return sum(counts) / len(counts) if counts else 0.0


def summarise(outputs: list[list[str]], references: list[list[str]], prompts: list[list[str]]) -> dict:
    return {
        "BLEU": bleu(outputs, references),
        "chrF": chrf(outputs, references),
        "distinct-2": distinct_n(outputs, 2),
        "copied lines": copy_rate(outputs, prompts),
        "words per line": words_per_line(outputs),
    }
