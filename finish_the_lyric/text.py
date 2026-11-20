"""Text helpers shared by data preparation, generation and evaluation.

A song is handled as a list of lines. Everything in this module is plain Python, so it
can be tested without PyTorch or a model.
"""

import re
import unicodedata

_TAG_WORDS = r"(?:chorus|pre-?chorus|verse|bridge|intro|outro|hook|refrain|solo|instrumental|repeat)"

# Lines that only label a section, e.g. "[Chorus]", "(Chorus x2)", "Verse 1:", "Chorus".
_SECTION_TAG = re.compile(
    rf"^(?:\[[^\]]*\]|\(\s*{_TAG_WORDS}[^)]*\)|{_TAG_WORDS}(?:\s*\d+)?(?:\s*x\s*\d+)?\s*:?)$",
    re.IGNORECASE,
)

# Repeat markers at the end of a line, e.g. "Hey hey (x2)", "Hey hey [2x]", "Hey hey x3".
_REPEAT_MARK = re.compile(r"\s*(?:[\[(]\s*(?:x\s*\d+|\d+\s*x)\s*[\])]|\s[x×]\s?\d+)$", re.IGNORECASE)

# A sentence end followed by more text: used to split lyrics stored without line breaks.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "…": "..."})


def lyric_lines(raw: str) -> list[str]:
    """Split raw lyrics into clean, non-empty lines.

    Section labels such as "[Chorus]" and repeat markers such as "(x2)" are removed,
    whitespace is tidied and curly quotes are straightened. Blank lines between verses
    are dropped.

    Some copies of the dataset store each song on a single line, with the original lines
    joined by full stops. When the text has no line breaks, it is split after each full
    stop, question mark or exclamation mark, and the joining full stop is removed.
    """
    text = unicodedata.normalize("NFKC", str(raw)).translate(_QUOTES)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" in text:
        candidates = text.split("\n")
    else:
        candidates = [_drop_final_full_stop(part) for part in _SENTENCE_END.split(text)]

    lines = []
    for line in candidates:
        line = re.sub(r"\s+", " ", line).strip()
        if not line or _SECTION_TAG.match(line):
            continue
        line = _REPEAT_MARK.sub("", line).strip()
        if line:
            lines.append(line)
    return lines


def _drop_final_full_stop(part: str) -> str:
    part = part.strip()
    if part.endswith(".") and not part.endswith(".."):
        part = part[:-1]
    return part


def split_song(lines: list[str], n_ending: int) -> tuple[list[str], list[str]]:
    """Split a song into the lines the model sees (prompt) and the lines it must write."""
    if n_ending < 1 or len(lines) <= n_ending:
        raise ValueError(f"need more than {n_ending} lines to hold back {n_ending}")
    return lines[:-n_ending], lines[-n_ending:]


def complete_lines(text: str) -> list[str]:
    """Non-empty lines of generated text that have been finished with a line break."""
    return [line.strip() for line in text.split("\n")[:-1] if line.strip()]


def words(text: str) -> list[str]:
    """Lower-case words, ignoring punctuation (apostrophes are kept: "don't")."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.translate(_QUOTES).lower())


def normalise(text: str) -> str:
    """A comparison key: lower-case words separated by single spaces."""
    return " ".join(words(text))
