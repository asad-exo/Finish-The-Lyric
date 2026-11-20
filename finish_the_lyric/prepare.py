"""Build the train, validation and test sets from the Kaggle lyrics dataset.

    python -m finish_the_lyric.prepare --artists data/raw/artists-data.csv --lyrics data/raw/lyrics-data.csv

Writes train.jsonl, val.jsonl, test.jsonl and stats.json to data/processed/. Each line of
the JSONL files is one song: {"artist", "title", "genre", "lines"}.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .text import lyric_lines, normalise
from .utils import write_json, write_jsonl

# The dataset has been published in two versions with different language columns.
_LANGUAGE_COLUMNS = {"language": "en", "Idiom": "ENGLISH"}


def load_artists(path: str | Path, genre: str, min_popularity: float) -> pd.DataFrame:
    """Artists whose main genre (the first one listed) contains `genre`."""
    artists = pd.read_csv(path)
    main_genre = artists["Genres"].fillna("").str.split(";").str[0].str.strip()
    keep = main_genre.str.contains(genre, case=False, regex=False) & (
        artists["Popularity"].fillna(0) > min_popularity
    )
    selected = artists.loc[keep, ["Artist", "Link"]].copy()
    selected["genre"] = main_genre[keep]
    return selected


def load_english_lyrics(path: str | Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    language_column = next((c for c in _LANGUAGE_COLUMNS if c in header), None)
    if language_column is None:
        raise ValueError(f"{path} has no 'language' or 'Idiom' column; is it lyrics-data.csv?")
    lyrics = pd.read_csv(path, usecols=["ALink", "SName", "Lyric", language_column])
    english = lyrics[language_column] == _LANGUAGE_COLUMNS[language_column]
    return lyrics[english].dropna(subset=["Lyric"]).drop(columns=language_column)


def title_key(title: str) -> str:
    """Song title without version notes: "Song (Live)" and "Song - Remastered" become "song"."""
    title = re.sub(r"[(\[].*?[)\]]", " ", str(title))
    title = re.sub(r"\s-\s.*$", " ", title)
    return normalise(title)


def build(
    artists_path: str | Path,
    lyrics_path: str | Path,
    out_dir: str | Path,
    genre: str = "rock",
    min_popularity: float = 5,
    min_lines: int = 8,
    max_words: int = 500,
    test_songs: int = 500,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    stats = {}
    artists = load_artists(artists_path, genre, min_popularity)
    stats["artists selected"] = len(artists)

    songs = load_english_lyrics(lyrics_path).merge(artists, left_on="ALink", right_on="Link")
    stats["English songs by those artists"] = len(songs)

    songs["lines"] = songs["Lyric"].map(lyric_lines)
    n_lines = songs["lines"].map(len)
    n_words = songs["lines"].map(lambda lines: sum(len(line.split()) for line in lines))
    songs = songs[(n_lines >= min_lines) & (n_words <= max_words)].copy()
    stats[f"with {min_lines}+ lines and up to {max_words} words"] = len(songs)

    # The same song often appears more than once (live versions, re-releases, covers).
    # Duplicates are removed before splitting, so no test song has a copy in training.
    songs["text_key"] = songs["lines"].map(lambda lines: normalise(" ".join(lines)))
    titles = songs["SName"].fillna("").map(title_key)
    songs["title_key"] = np.where(
        titles != "", songs["Artist"].str.lower() + "|" + titles, "untitled|" + songs.index.astype(str)
    )
    songs = songs.drop_duplicates("text_key").drop_duplicates("title_key")
    stats["after removing duplicates"] = len(songs)

    order = np.random.default_rng(seed).permutation(len(songs))
    songs = songs.iloc[order]
    n_test = min(test_songs, len(songs) // 5)
    n_val = round((len(songs) - n_test) * val_fraction)
    splits = {
        "test": songs.iloc[:n_test],
        "val": songs.iloc[n_test : n_test + n_val],
        "train": songs.iloc[n_test + n_val :],
    }

    out_dir = Path(out_dir)
    for name, part in splits.items():
        records = [
            {"artist": row.Artist, "title": row.SName, "genre": row.genre, "lines": row.lines}
            for row in part.itertuples()
        ]
        write_jsonl(out_dir / f"{name}.jsonl", records)
        stats[f"{name} songs"] = len(records)
        stats[f"{name} lines"] = int(part["lines"].map(len).sum())

    write_json(out_dir / "stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artists", default="data/raw/artists-data.csv")
    parser.add_argument("--lyrics", default="data/raw/lyrics-data.csv")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--genre", default="rock", help="text to look for in each artist's main genre")
    parser.add_argument("--min-popularity", type=float, default=5)
    parser.add_argument("--min-lines", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=500)
    parser.add_argument("--test-songs", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = build(
        args.artists, args.lyrics, args.out,
        genre=args.genre, min_popularity=args.min_popularity, min_lines=args.min_lines,
        max_words=args.max_words, test_songs=args.test_songs, val_fraction=args.val_fraction,
        seed=args.seed,
    )
    width = max(len(key) for key in stats)
    for key, value in stats.items():
        print(f"{key:<{width}}  {value:>9,}")
    print(f"\nWritten to {args.out}/")


if __name__ == "__main__":
    main()
