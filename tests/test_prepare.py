import pandas as pd
import pytest

from finish_the_lyric.prepare import build, title_key
from finish_the_lyric.utils import read_jsonl


def song(n, word):
    return "\n".join(f"{word} line number {i}" for i in range(n))


@pytest.fixture
def raw_files(tmp_path):
    artists = pd.DataFrame({
        "Artist": ["Rockers", "Popsters", "Quiet Rock", "Hard Ones"],
        "Genres": ["Rock; Pop", "Pop; Rock", "Rock", "Hard Rock; Heavy Metal"],
        "Songs": [10, 10, 10, 10],
        "Popularity": [9.0, 9.0, 1.0, 7.5],
        "Link": ["/rockers/", "/popsters/", "/quiet-rock/", "/hard-ones/"],
    })
    rows = [("/rockers/", f"Song {i}", song(10, f"rock{i}"), "en") for i in range(40)]
    rows += [
        ("/rockers/", "Song 1 (Live)", song(10, "different live words"), "en"),  # same title, live version
        ("/hard-ones/", "Cover", song(10, "rock2"), "en"),                       # same lyrics as Song 2
        ("/hard-ones/", "Short", song(3, "short"), "en"),                        # too short
        ("/hard-ones/", "Portuguese", song(10, "canção"), "pt"),                 # not English
        ("/popsters/", "Pop song", song(10, "pop"), "en"),                       # main genre is Pop
        ("/quiet-rock/", "Unpopular", song(10, "quiet"), "en"),                  # popularity too low
    ]
    lyrics = pd.DataFrame(rows, columns=["ALink", "SName", "Lyric", "language"])
    lyrics.insert(2, "SLink", "/x/")
    artists_path, lyrics_path = tmp_path / "artists-data.csv", tmp_path / "lyrics-data.csv"
    artists.to_csv(artists_path, index=False)
    lyrics.to_csv(lyrics_path, index=False)
    return artists_path, lyrics_path, tmp_path / "processed"


def test_build_filters_deduplicates_and_splits(raw_files):
    artists_path, lyrics_path, out = raw_files
    stats = build(artists_path, lyrics_path, out, test_songs=5, val_fraction=0.1)

    assert stats["artists selected"] == 2
    assert stats["English songs by those artists"] == 43
    assert stats["after removing duplicates"] == 40
    splits = {name: read_jsonl(out / f"{name}.jsonl") for name in ("train", "val", "test")}
    assert [len(splits[n]) for n in ("test", "val", "train")] == [5, 4, 31]

    keys = ["\n".join(s["lines"]) for part in splits.values() for s in part]
    assert len(keys) == len(set(keys))
    titles = {s["title"] for part in splits.values() for s in part}
    assert {"Short", "Portuguese", "Pop song", "Unpopular", "Song 1 (Live)"}.isdisjoint(titles)


def test_build_reads_the_older_dataset_version(raw_files):
    artists_path, lyrics_path, out = raw_files
    lyrics = pd.read_csv(lyrics_path)
    lyrics["Idiom"] = lyrics.pop("language").map({"en": "ENGLISH", "pt": "PORTUGUESE"})
    lyrics.to_csv(lyrics_path, index=False)
    assert build(artists_path, lyrics_path, out, test_songs=5)["after removing duplicates"] == 40


def test_title_key_ignores_version_notes():
    assert title_key("Hey Jude (Live)") == title_key("Hey Jude - Remastered 2009") == "hey jude"
