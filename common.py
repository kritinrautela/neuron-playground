"""Shared helpers: model loading, chat formatting, hooks, steering, generation."""

import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
if (ROOT / "hf_cache" / "hub" / "models--LiquidAI--LFM2.5-2.6B" / "snapshots").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are already downloaded
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

MODEL_ID = "LiquidAI/LFM2.5-2.6B"
RESULTS = ROOT / "results"
RESID_NORMS_PATH = RESULTS / "resid_norms.json"
CANDIDATES_PATH = RESULTS / "candidates.json"

NEUTRAL_SENTENCES = [
    "The train leaves the station at half past nine every morning.",
    "Water boils at a lower temperature at high altitude.",
    "She planted tomatoes and basil in the small garden behind the house.",
    "The library is closed on public holidays but open on weekends.",
    "A typical week has five working days and two days of rest.",
    "The recipe calls for two cups of flour and a pinch of salt.",
]


def load_model():
    """Load model + tokenizer in bf16 on CUDA or MPS, falling back to CPU if neither works."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.eval()
    gpus = [d for d, ok in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())) if ok]
    for dev in gpus:
        try:
            model.to(dev)
            with torch.no_grad():
                model(torch.tensor([[tok.bos_token_id, 100]], device=dev))
            return model, tok
        except Exception as e:
            print(f"[warn] {dev} failed ({e}); trying the next device")
    model.to("cpu")
    return model, tok


def layers(model):
    return model.model.layers


def down_proj(model, layer):
    return layers(model)[layer].feed_forward.w2


# LFM2.5-2.6B always reasons inside <think>...</think> first. The template already opens
# the tag; closing it straight away skips the reasoning so 200 tokens go to the answer.
SKIP_THINKING = "\n</think>\n\n"


def chat_ids(tok, messages):
    """Token ids for a conversation, ending at the start of the assistant's answer."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok(text + SKIP_THINKING, add_special_tokens=False).input_ids


def response_span(tok, user, reply):
    """Token ids for a user/assistant exchange plus the index where the reply starts."""
    prompt = chat_ids(tok, [{"role": "user", "content": user}])
    answer = tok(reply + "<|im_end|>", add_special_tokens=False).input_ids
    return prompt + answer, len(prompt)


# ---------------------------------------------------------------- steering

class Steerer:
    """Adds a fixed amount to chosen neurons at the down-projection input, every position."""

    def __init__(self, model):
        self.model = model
        self.steers = {}  # (layer, neuron) -> dose
        self.enabled = True
        self.resid_norms = None
        self.handles = []
        for i in range(len(layers(model))):
            self.handles.append(down_proj(model, i).register_forward_pre_hook(self._make_hook(i)))

    def _make_hook(self, layer):
        def hook(module, args):
            if not self.enabled:
                return None
            mine = [(n, d) for (l, n), d in self.steers.items() if l == layer]
            if not mine:
                return None
            x = args[0].clone()
            for n, dose in mine:
                x[..., n] += self.amount(layer, n, dose)
            return (x,) + tuple(args[1:])
        return hook

    def amount(self, layer, neuron, dose):
        col_norm = down_proj(self.model, layer).weight[:, neuron].float().norm().item()
        return dose * self.resid_norms[layer] / col_norm

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def compute_resid_norms(model, tok):
    """Median L2 norm of the residual leaving each layer, over tokens 1: of neutral text."""
    per_layer = [[] for _ in layers(model)]
    hooks = []
    for i, layer in enumerate(layers(model)):
        def hook(module, args, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            per_layer[i].append(h[0, 1:].float().norm(dim=-1).cpu())
        hooks.append(layer.register_forward_hook(hook))
    try:
        for s in NEUTRAL_SENTENCES:
            ids = tok(s, return_tensors="pt").input_ids
            if ids[0, 0] != tok.bos_token_id:
                ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1)
            model(ids.to(model.device))
    finally:
        for h in hooks:
            h.remove()
    return [torch.cat(v).median().item() for v in per_layer]


def get_resid_norms(model, tok):
    if RESID_NORMS_PATH.exists():
        return json.loads(RESID_NORMS_PATH.read_text())
    norms = compute_resid_norms(model, tok)
    RESULTS.mkdir(exist_ok=True)
    RESID_NORMS_PATH.write_text(json.dumps(norms, indent=1))
    return norms


@torch.no_grad()
def generate(model, tok, messages, max_new_tokens=200):
    ids = torch.tensor([chat_ids(tok, messages)], device=model.device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
