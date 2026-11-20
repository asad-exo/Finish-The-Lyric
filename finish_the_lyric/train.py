"""Fine-tune GPT-2 on the training songs.

    python -m finish_the_lyric.train --data data/processed --out models/finish-the-lyric

Songs are joined into one stream, each followed by an end-of-text token, and cut into
equal blocks. The model learns to predict every next token. After each epoch the
validation loss is checked, and the best model so far is saved to --out.
"""

import argparse
import contextlib
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from .modeling import pick_device, song_text
from .utils import read_jsonl, set_seed, write_json


def pack(songs: list[dict], tokenizer, block_size: int) -> torch.Tensor:
    """Token blocks of shape (n_blocks, block_size). Leftover tokens at the end are dropped."""
    ids = [tokenizer.eos_token_id]
    for song in songs:
        ids.extend(tokenizer(song_text(song["lines"]))["input_ids"])
        ids.append(tokenizer.eos_token_id)
    n_blocks = len(ids) // block_size
    if n_blocks == 0:
        raise ValueError(f"only {len(ids)} tokens: not enough for one block of {block_size}")
    return torch.tensor(ids[: n_blocks * block_size], dtype=torch.long).view(n_blocks, block_size)


def mixed_precision(device: torch.device):
    """Use bfloat16 or float16 on a CUDA GPU; full precision elsewhere."""
    if device.type != "cuda":
        return contextlib.nullcontext, None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return (lambda: torch.autocast(device_type="cuda", dtype=dtype)), dtype


@torch.no_grad()
def mean_loss(model, blocks: torch.Tensor, batch_size: int, device, autocast) -> float:
    model.eval()
    total = 0.0
    for batch in DataLoader(blocks, batch_size=batch_size):
        batch = batch.to(device)
        with autocast():
            total += model(input_ids=batch, labels=batch).loss.item() * len(batch)
    model.train()
    return total / len(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="models/finish-the-lyric")
    parser.add_argument("--model", default="gpt2", help="starting model, from the Hub or a local folder")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4, help="batches per optimizer step")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.06, help="share of steps used for warm-up")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1, help="epochs without improvement before stopping")
    parser.add_argument("--max-train-songs", type=int, help="use only this many songs, for a quick run")
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    autocast, dtype = mixed_precision(device)
    print(f"Device: {device}" + (f" ({dtype})" if dtype else ""))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.train()

    train_songs = read_jsonl(Path(args.data) / "train.jsonl")[: args.max_train_songs]
    val_songs = read_jsonl(Path(args.data) / "val.jsonl")
    train_blocks = pack(train_songs, tokenizer, args.block_size)
    val_blocks = pack(val_songs, tokenizer, args.block_size)
    print(f"{len(train_songs):,} training songs -> {len(train_blocks):,} blocks of {args.block_size} tokens")
    print(f"{len(val_songs):,} validation songs -> {len(val_blocks):,} blocks")

    # No weight decay on biases and layer-norm weights.
    params = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for _, p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
            {"params": [p for _, p in params if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    loader = DataLoader(train_blocks, batch_size=args.batch_size, shuffle=True)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, round(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    out = Path(args.out)
    val_loss = mean_loss(model, val_blocks, args.batch_size, device, autocast)
    log = [{"epoch": 0, "train_loss": None, "val_loss": val_loss, "val_perplexity": math.exp(val_loss)}]
    print(f"Before fine-tuning: validation loss {val_loss:.3f}, perplexity {math.exp(val_loss):.1f}")
    best, epochs_without_improvement = math.inf, 0

    for epoch in range(1, args.epochs + 1):
        started, running = time.time(), 0.0
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")):
            batch = batch.to(device)
            with autocast():
                loss = model(input_ids=batch, labels=batch).loss
            scaler.scale(loss / args.grad_accum).backward()
            running += loss.item()
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        train_loss = running / len(loader)
        val_loss = mean_loss(model, val_blocks, args.batch_size, device, autocast)
        minutes = (time.time() - started) / 60
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "val_perplexity": math.exp(val_loss), "minutes": round(minutes, 1)})
        print(f"Epoch {epoch}: train loss {train_loss:.3f}, validation loss {val_loss:.3f} "
              f"(perplexity {math.exp(val_loss):.1f}), {minutes:.1f} min")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "training_log.json", {"settings": vars(args), "epochs": log})

        if val_loss < best:
            best, epochs_without_improvement = val_loss, 0
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print(f"  saved to {out}/")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > args.patience:
                print("  no improvement: stopping early")
                break

    print(f"Best validation loss {best:.3f} (perplexity {math.exp(best):.1f})")


if __name__ == "__main__":
    main()
