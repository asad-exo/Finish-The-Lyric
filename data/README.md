# Data

Download the [*Song lyrics from 6 musical genres*](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres) dataset from Kaggle and put its two files here:

```
data/raw/artists-data.csv
data/raw/lyrics-data.csv
```

`python -m finish_the_lyric.prepare` reads them and writes `data/processed/`: `train.jsonl`, `val.jsonl` and `test.jsonl` (one song per line, as a list of lyric lines) and `stats.json` (how many songs each filter kept).

Both published versions of the dataset work: the newer one marks the language in a `language` column (`en`), the older one in `Idiom` (`ENGLISH`). The data isn't included in this repository because of its size (over 400 MB).
