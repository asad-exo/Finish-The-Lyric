import pytest

from finish_the_lyric.text import complete_lines, lyric_lines, normalise, split_song, words


def test_lines_are_cleaned():
    raw = "[Verse 1]\nI walk the line  \n\nChorus:\nHey, hey (x2)\n(Chorus x2)\nDon’t stop now [2x]\r\nRoute 66 x3\n"
    assert lyric_lines(raw) == ["I walk the line", "Hey, hey", "Don't stop now", "Route 66"]


def test_lines_that_only_mention_a_section_word_are_kept():
    assert lyric_lines("Chorus of angels sing\nVerse after verse") == ["Chorus of angels sing", "Verse after verse"]


def test_lyrics_without_line_breaks_are_split_at_sentence_ends():
    raw = "I feel so extraordinary. Something's got a hold on me? I get this feeling. Wait..."
    assert lyric_lines(raw) == ["I feel so extraordinary", "Something's got a hold on me?", "I get this feeling", "Wait..."]


def test_split_song_holds_back_the_ending():
    prompt, ending = split_song(["a", "b", "c", "d", "e"], 2)
    assert prompt == ["a", "b", "c"]
    assert ending == ["d", "e"]
    with pytest.raises(ValueError):
        split_song(["a", "b"], 2)


def test_complete_lines_ignores_an_unfinished_last_line():
    assert complete_lines("one\n\ntwo\nthree is cut o") == ["one", "two"]
    assert complete_lines("no line break yet") == []


def test_words_and_normalise():
    assert words("Don’t STOP, believin'!") == ["don't", "stop", "believin"]
    assert normalise("  Hey,   HEY! ") == "hey hey"
