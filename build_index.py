"""One-time scan for the explorer: logit lens, 2D layouts and top-firing examples per neuron.

Writes index/index.pt (~100 MB). Takes ~15-25 min on an M2 Air.
"""

import gzip
import json
import sys
import time
import urllib.request

import torch

from common import ROOT, down_proj, layers, load_model, response_span
from find_neurons import PAIRS

INDEX = ROOT / "index"
CORPUS_GZ = INDEX / "c4-validation-00000.json.gz"
CORPUS_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz"
N_DOCS = 3000
SEQ = 64           # tokens per web snippet (after BOS)
BATCH = 16
TOP_K = 8          # top-firing examples kept per neuron
LENS_K = 10        # logit-lens tokens kept per neuron and direction
GRID = 104         # 104 x 104 >= 10752 for the "index" layout


def load_corpus(tok):
    """Web snippets from C4 plus the chat-formatted hostile/kind replies."""
    if not CORPUS_GZ.exists():
        print("downloading a 40 MB sample of web text (C4)...")
        INDEX.mkdir(exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_GZ)
    seqs = []
    with gzip.open(CORPUS_GZ, "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"], add_special_tokens=False).input_ids[:SEQ]
            if len(ids) >= 16:
                seqs.append([tok.bos_token_id] + ids)
            if len(seqs) >= N_DOCS:
                break
    for user, bad, good in PAIRS:
        for reply in (bad, good):
            seqs.append(response_span(tok, user, reply)[0][: SEQ + 1])
    return seqs


@torch.no_grad()
def logit_lens(model):
    """Top tokens each neuron's output direction pushes up / down -> int32 [L, N, K] x2."""
    unembed = model.lm_head.weight                      # [V, D]
    gain = model.model.embedding_norm.weight            # final RMSNorm gain
    ups, downs = [], []
    for i in range(len(layers(model))):
        w = down_proj(model, i).weight                  # [D, N]
        up_l, down_l = [], []
        for s in range(0, w.shape[1], 2048):
            cols = w[:, s:s + 2048].T * gain            # RMS scale doesn't change the ranking
            logits = (cols @ unembed.T).float()
            up_l.append(logits.topk(LENS_K, dim=1).indices.int().cpu())
            down_l.append((-logits).topk(LENS_K, dim=1).indices.int().cpu())
        ups.append(torch.cat(up_l))
        downs.append(torch.cat(down_l))
        print(f"\rlogit lens layer {i + 1}/{len(layers(model))}", end="", flush=True)
    print()
    return torch.stack(ups), torch.stack(downs)


@torch.no_grad()
def write_layouts(model):
    """2D positions per neuron in [-1, 1]: by index (grid) and by write direction (PCA)."""
    n = down_proj(model, 0).weight.shape[1]
    idx = torch.arange(n)
    grid = torch.stack([idx % GRID, idx // GRID], dim=1).float()
    grid = grid / (GRID - 1) * 2 - 1
    pca = []
    for i in range(len(layers(model))):
        cols = down_proj(model, i).weight.T.float().cpu()
        cols = cols / cols.norm(dim=1, keepdim=True).clamp_min(1e-6)
        _, _, v = torch.pca_lowrank(cols, q=2, center=True)
        xy = (cols - cols.mean(0)) @ v
        # spread evenly: rank-transform each axis so dots fill the plane
        xy = torch.stack([c.argsort().argsort().float() for c in xy.T], dim=1)
        pca.append(xy / (n - 1) * 2 - 1)
    return grid, torch.stack(pca)


@torch.no_grad()
def scan(model, tok, seqs):
    """Running top-K max activations (+ where) and mean/std per neuron over the corpus."""
    n_layers = len(layers(model))
    n = down_proj(model, 0).weight.shape[1]
    dev = model.device
    top_val = torch.full((n_layers, TOP_K, n), -1e9, device=dev)
    top_pos = torch.zeros((n_layers, TOP_K, n), dtype=torch.long, device=dev)
    total = torch.zeros((n_layers, n), device=dev)
    total_sq = torch.zeros((n_layers, n), device=dev)
    count = 0
    width = SEQ + 1
    tokens = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.int32)
    for j, s in enumerate(seqs):
        tokens[j, : len(s)] = torch.tensor(s)
    state = {}

    def make_hook(i):
        def hook(module, args):
            acts = args[0][:, 1:].float()               # skip token 0 (attention sink)
            mask = state["mask"]
            flat = acts.reshape(-1, n)[mask]
            pos = state["pos"][mask]
            total[i] += flat.sum(0)
            total_sq[i] += (flat * flat).sum(0)
            k = min(TOP_K, flat.shape[0])
            v, ix = flat.topk(k, dim=0)
            vals = torch.cat([top_val[i], v])
            poss = torch.cat([top_pos[i], pos[ix]])
            keep = vals.topk(TOP_K, dim=0).indices
            top_val[i] = vals.gather(0, keep)
            top_pos[i] = poss.gather(0, keep)
        return hook

    hooks = [down_proj(model, i).register_forward_pre_hook(make_hook(i)) for i in range(n_layers)]
    t0 = time.time()
    try:
        for b in range(0, len(seqs), BATCH):
            batch = tokens[b:b + BATCH].long()
            valid = batch != tok.pad_token_id
            pos = (torch.arange(b, b + len(batch))[:, None] * width + torch.arange(width)[None, :])
            state["mask"] = valid[:, 1:].reshape(-1).to(dev)
            state["pos"] = pos[:, 1:].reshape(-1).to(dev)
            count += int(valid[:, 1:].sum())
            model(batch.to(dev), attention_mask=valid.long().to(dev))
            done = min(b + BATCH, len(seqs))
            eta = (time.time() - t0) / done * (len(seqs) - done)
            print(f"\rscan {done}/{len(seqs)} sequences  eta {eta / 60:.1f} min", end="", flush=True)
    finally:
        for h in hooks:
            h.remove()
    print()
    mean = total / count
    std = (total_sq / count - mean * mean).clamp_min(0).sqrt()
    return tokens, top_val.cpu().half(), top_pos.cpu().int(), mean.cpu(), std.cpu()


def main():
    t0 = time.time()
    model, tok = load_model()
    seqs = load_corpus(tok)
    print(f"corpus: {len(seqs)} sequences")
    lens_up, lens_down = logit_lens(model)
    grid, pca = write_layouts(model)
    tokens, top_val, top_pos, mean, std = scan(model, tok, seqs)
    torch.save({
        "seq_len": SEQ + 1,
        "tokens": tokens,
        "top_val": top_val, "top_pos": top_pos,
        "mean": mean, "std": std,
        "lens_up": lens_up, "lens_down": lens_down,
        "layout_index": grid, "layout_writes": pca,
    }, INDEX / "index.pt")
    print(f"saved {INDEX / 'index.pt'} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
