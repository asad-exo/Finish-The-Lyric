"""Loading a model, continuing a song with it, and scoring a song's real ending."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .text import complete_lines


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(name_or_path: str, device: torch.device):
    """A causal language model and its tokenizer, from the Hugging Face Hub or a local folder."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForCausalLM.from_pretrained(name_or_path).to(device).eval()
    return model, tokenizer


def song_text(lines: list[str]) -> str:
    return "\n".join(lines)


def prompt_ids(tokenizer, prompt_lines: list[str], max_tokens: int) -> list[int]:
    """Token ids for the song so far, ending with a line break.

    In training, every song starts after an end-of-text token, so the prompt does too.
    Long prompts keep their last `max_tokens` tokens: the lines nearest the ending matter most.
    """
    ids = [tokenizer.eos_token_id] + tokenizer(song_text(prompt_lines) + "\n")["input_ids"]
    return ids[-max_tokens:]


@dataclass
class Sampling:
    """Settings for nucleus sampling."""

    top_p: float = 0.9
    temperature: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 100


class _StopAfterLines(StoppingCriteria):
    """Stops generating once the new text contains `n_lines` finished lines."""

    def __init__(self, tokenizer, prompt_length: int, n_lines: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.n_lines = n_lines

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.prompt_length :], skip_special_tokens=True)
        done = len(complete_lines(text)) >= self.n_lines
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def continue_lyric(
    model,
    tokenizer,
    prompt_lines: list[str],
    n_lines: int = 4,
    sampling: Sampling = Sampling(),
    max_prompt_tokens: int = 384,
    seed: int | None = None,
) -> list[str]:
    """Write up to `n_lines` new lines that follow `prompt_lines`."""
    ids = prompt_ids(tokenizer, prompt_lines, max_prompt_tokens)
    input_ids = torch.tensor([ids], device=model.device)
    if seed is not None:
        torch.manual_seed(seed)
    output = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=True,
        top_p=sampling.top_p,
        temperature=sampling.temperature,
        repetition_penalty=sampling.repetition_penalty,
        max_new_tokens=sampling.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([_StopAfterLines(tokenizer, len(ids), n_lines)]),
    )
    new_tokens = output[0, len(ids) :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if tokenizer.eos_token_id in new_tokens.tolist():
        # The model ended the song: everything it wrote counts, finished with a line break or not.
        lines = [line.strip() for line in text.split("\n") if line.strip()]
    else:
        # Otherwise drop a last line that was cut off by the token limit, if there are others.
        lines = complete_lines(text) or [line.strip() for line in text.split("\n") if line.strip()]
    return lines[:n_lines]


@torch.no_grad()
def ending_log_likelihood(model, tokenizer, prompt_lines, ending_lines, max_prompt_tokens: int = 384):
    """Negative log-likelihood (in nats) of a song's real ending, given the lines before it.

    Returns (total over the ending's tokens, number of tokens). Only the ending is scored;
    the prompt is context. exp(sum of totals / sum of tokens) is the perplexity of the
    real endings: lower means the model expected what the songwriter actually wrote.
    """
    context = prompt_ids(tokenizer, prompt_lines, max_prompt_tokens)
    ending = tokenizer(song_text(ending_lines))["input_ids"]
    input_ids = torch.tensor([context + ending], device=model.device)
    labels = torch.tensor([[-100] * len(context) + ending], device=model.device)
    mean_loss = model(input_ids=input_ids, labels=labels).loss.item()
    return mean_loss * len(ending), len(ending)
