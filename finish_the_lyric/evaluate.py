"""Finish held-out songs and score the results.

    python -m finish_the_lyric.evaluate --data data/processed --model models/finish-the-lyric

The last --lines lines of each test song are held back. Each system writes its own
ending from the lines before, and the output is compared with the real ending.

Systems:
  Fine-tuned GPT-2       the model trained with finish_the_lyric.train (--model)
  GPT-2, not fine-tuned  the starting model (--base-model), to show what fine-tuning adds
  Repeat last lines      copies the last lines of the prompt; songs often end on a repeated chorus

Writes metrics.json, metrics.md and samples.csv to --out.
"""

import argparse
import csv
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from . import metrics
from .modeling import Sampling, continue_lyric, ending_log_likelihood, load, pick_device
from .text import split_song
from .utils import read_jsonl, set_seed, write_json

REAL = "Real endings"


def run_model(path, cases, n_lines, sampling, max_prompt_tokens, device, seed):
    model, tokenizer = load(path, device)
    outputs, nll, tokens = [], 0.0, 0
    for i, (prompt, ending) in enumerate(tqdm(cases, desc=path)):
        outputs.append(continue_lyric(model, tokenizer, prompt, n_lines, sampling, max_prompt_tokens, seed=seed + i))
        total, count = ending_log_likelihood(model, tokenizer, prompt, ending, max_prompt_tokens)
        nll += total
        tokens += count
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs, math.exp(nll / tokens)


def markdown_table(rows: list[dict]) -> str:
    columns = ["system", "BLEU", "chrF", "ending perplexity", "distinct-2", "copied lines", "words per line"]
    formats = {"BLEU": "{:.2f}", "chrF": "{:.1f}", "ending perplexity": "{:.1f}",
               "distinct-2": "{:.2f}", "copied lines": "{:.0%}", "words per line": "{:.1f}"}
    lines = ["| " + " | ".join(columns) + " |", "| --- |" + " ---: |" * (len(columns) - 1)]
    for row in rows:
        cells = [row["system"]] + [
            formats[c].format(row[c]) if row.get(c) is not None else "–" for c in columns[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--model", help="fine-tuned model folder (skip to evaluate only the baselines)")
    parser.add_argument("--base-model", default="gpt2", help="model without fine-tuning; 'none' to skip")
    parser.add_argument("--out", default="outputs/evaluation")
    parser.add_argument("--lines", type=int, default=4, help="lines to hold back and generate")
    parser.add_argument("--min-prompt-lines", type=int, default=4)
    parser.add_argument("--max-songs", type=int, default=500)
    parser.add_argument("--max-prompt-tokens", type=int, default=384)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    songs = [
        s for s in read_jsonl(Path(args.data) / "test.jsonl")
        if len(s["lines"]) >= args.lines + args.min_prompt_lines
    ][: args.max_songs]
    cases = [split_song(s["lines"], args.lines) for s in songs]
    prompts = [prompt for prompt, _ in cases]
    references = [ending for _, ending in cases]
    print(f"{len(cases)} test songs; generating the last {args.lines} lines of each")

    sampling = Sampling(top_p=args.top_p, temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty)
    systems = {}
    perplexity = {}
    model_runs = [("Fine-tuned GPT-2", args.model), ("GPT-2, not fine-tuned", args.base_model)]
    for name, path in model_runs:
        if path and path.lower() != "none":
            systems[name], perplexity[name] = run_model(
                path, cases, args.lines, sampling, args.max_prompt_tokens, device, args.seed
            )
    systems["Repeat last lines"] = [prompt[-args.lines :] for prompt in prompts]

    rows = []
    for name, outputs in systems.items():
        rows.append({"system": name, **metrics.summarise(outputs, references, prompts),
                     "ending perplexity": perplexity.get(name)})
    rows.append({"system": REAL, "BLEU": None, "chrF": None, "ending perplexity": None,
                 "distinct-2": metrics.distinct_n(references), "copied lines": metrics.copy_rate(references, prompts),
                 "words per line": metrics.words_per_line(references)})

    out = Path(args.out)
    write_json(out / "metrics.json", {"settings": vars(args), "songs": len(cases), "results": rows})
    table = markdown_table(rows)
    (out / "metrics.md").write_text(table + "\n", encoding="utf-8")
    with open(out / "samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artist", "title", "last prompt lines", REAL, *systems])
        for i, song in enumerate(songs):
            writer.writerow([song["artist"], song["title"], "\n".join(prompts[i][-4:]), "\n".join(references[i]),
                             *("\n".join(outputs[i]) for outputs in systems.values())])

    print("\n" + table)
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
