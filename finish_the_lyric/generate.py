"""Finish your own lyrics.

    python -m finish_the_lyric.generate --model models/finish-the-lyric "First line / Second line / Third line"

Separate lines with " / " or line breaks. With no text argument, the lyrics are read
from standard input, so a file can be piped in.
"""

import argparse
import sys

from .modeling import Sampling, continue_lyric, load, pick_device
from .text import lyric_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lyrics", nargs="?", help="the song so far (default: read from standard input)")
    parser.add_argument("--model", default="models/finish-the-lyric")
    parser.add_argument("--lines", type=int, default=4, help="how many lines to write")
    parser.add_argument("--samples", type=int, default=3, help="how many different endings to write")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    raw = args.lyrics if args.lyrics is not None else sys.stdin.read()
    prompt = lyric_lines(raw.replace(" / ", "\n"))
    if not prompt:
        parser.error("no lyrics given")

    model, tokenizer = load(args.model, pick_device(args.device))
    sampling = Sampling(top_p=args.top_p, temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty)
    for i in range(args.samples):
        seed = None if args.seed is None else args.seed + i
        lines = continue_lyric(model, tokenizer, prompt, args.lines, sampling, seed=seed)
        print(f"--- Ending {i + 1}")
        print("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
