"""Neuron explorer server: 3D neuron map, neuron details, steering and a Claude side panel.

    python explorer.py            # then open http://127.0.0.1:8765
"""

import base64
import json
import mimetypes
import queue
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

from common import ROOT, Steerer, chat_ids, down_proj, get_resid_norms, layers, load_model

PORT = 8765
STATIC = ROOT / "explorer" / "static"
INDEX_PATH = ROOT / "index" / "index.pt"
AGENT_DIR = ROOT / "agent"
AGENT_MODEL = "sonnet"
STEER_LENS_PROMPTS = [
    "Tell me a little about yourself.",
    "What do you think about the weather today?",
    "Write one sentence about the ocean.",
]


HISTORY_MSGS = 6                                   # last 3 exchanges are remembered in a chat
PRESETS_PATH = ROOT / "results" / "presets.json"   # saved characters (named sets of steers)

class Explorer:
    def __init__(self):
        print("loading model...")
        self.model, self.tok = load_model()
        self.lock = threading.RLock()  # MPS is not thread-safe: one model call at a time
        self.steerer = Steerer(self.model)
        self.steerer.resid_norms = get_resid_norms(self.model, self.tok)
        self.n_layers = len(layers(self.model))
        self.n = down_proj(self.model, 0).weight.shape[1]
        self.types = list(self.model.config.layer_types)
        self.vocab = [self.tok.decode([i]) for i in range(len(self.tok))]
        self.last_trace = None  # (token strings, start index, [T, L, N] fp16)
        self.listeners = set()  # queues of live events for open browser tabs
        self.idx = torch.load(INDEX_PATH, weights_only=True) if INDEX_PATH.exists() else None
        if self.idx is not None:
            flat = self.idx["tokens"].reshape(-1)
            self.top_tok = flat[self.idx["top_pos"].long()]              # [L, K, N]
            self.act_max = self.idx["top_val"][:, 0].float().clamp_min(1e-4)  # [L, N]
        self._weights_for_web()
        print(f"index: {'loaded' if self.idx is not None else 'missing - run build_index.py'}")

    # ------------------------------------------------------------- weights

    @torch.no_grad()
    def _weights_for_web(self):
        """Unit read (gate) and write directions per neuron, for neuron-to-neuron links."""
        reads, writes = [], []
        for layer in layers(self.model):
            r = layer.feed_forward.w1.weight * layer.ffn_norm.weight   # [N, D]
            w = layer.feed_forward.w2.weight.T                         # [N, D]
            reads.append(r / r.norm(dim=1, keepdim=True).clamp_min(1e-6))
            writes.append(w / w.norm(dim=1, keepdim=True).clamp_min(1e-6))
        self.reads = torch.stack(reads)
        self.writes = torch.stack(writes)

    @torch.no_grad()
    def web(self, layer, neuron, k=10):
        """Upstream neurons that excite/inhibit this one; downstream ones it excites/inhibits."""
        def ranked(scores, layer_offset):
            flat = scores.flatten()
            out = []
            for sign in (1, -1):
                vals, ix = (sign * flat).topk(min(k, flat.numel()))
                out.append([self._ref(layer_offset + i // self.n, i % self.n, sign * v)
                            for v, i in zip(vals.tolist(), ix.tolist())])
            return out

        empty = [[], []]
        fed = ranked(self.writes[:layer] @ self.reads[layer, neuron], 0) if layer > 0 else empty
        feeds = (ranked(self.reads[layer + 1:] @ self.writes[layer, neuron], layer + 1)
                 if layer < self.n_layers - 1 else empty)
        return {"fed_by": fed[0], "held_back_by": fed[1],
                "pushes_toward": feeds[0], "pushes_away": feeds[1]}

    def _ref(self, layer, neuron, weight):
        ref = {"layer": layer, "neuron": neuron, "w": round(weight, 3)}
        if self.idx is not None:
            ref["hint"] = self.vocab[int(self.idx["lens_up"][layer, neuron, 0])]
        return ref

    # ------------------------------------------------------------- neuron view

    @torch.no_grad()
    def neuron(self, layer, neuron):
        out = {"layer": layer, "neuron": neuron, "type": self.types[layer],
               "web": self.web(layer, neuron)}
        if self.idx is None:
            return out
        out["lens_up"] = [self.vocab[int(t)] for t in self.idx["lens_up"][layer, neuron]]
        out["lens_down"] = [self.vocab[int(t)] for t in self.idx["lens_down"][layer, neuron]]
        out["mean"] = round(float(self.idx["mean"][layer, neuron]), 5)
        out["std"] = round(float(self.idx["std"][layer, neuron]), 5)
        out["max"] = round(float(self.act_max[layer, neuron]), 5)
        out["examples"] = self.examples(layer, neuron)
        if self.last_trace is not None:
            toks, start, acts = self.last_trace
            out["trace"] = {"tokens": toks[start:],
                            "acts": [round(float(a), 5) for a in acts[start:, layer, neuron]]}
        return out

    @torch.no_grad()
    def examples(self, layer, neuron, window=12):
        """Top-firing corpus snippets with this neuron's activation on every token."""
        seq_len = self.idx["seq_len"]
        pos = self.idx["top_pos"][layer, :, neuron].long()
        vals = self.idx["top_val"][layer, :, neuron].float()
        rows = (pos // seq_len).tolist()
        uniq = list(dict.fromkeys(rows))
        batch = self.idx["tokens"][uniq].long()
        mask = batch != self.tok.pad_token_id
        captured = {}
        hook = down_proj(self.model, layer).register_forward_pre_hook(
            lambda m, a: captured.__setitem__("x", a[0][..., neuron].float().cpu()))
        try:
            with self.lock:
                self.steerer.enabled = False
                self.model(batch.to(self.model.device), attention_mask=mask.long().to(self.model.device))
        finally:
            self.steerer.enabled = True
            hook.remove()
        out = []
        for p, v, r in zip(pos.tolist(), vals.tolist(), rows):
            if v <= -1e8:
                continue
            col = p % seq_len
            i = uniq.index(r)
            lo, hi = max(1, col - window), min(int(mask[i].sum()), col + 4)
            out.append({"act": round(v, 5), "at": col - lo,
                        "tokens": [self.vocab[int(t)] for t in batch[i, lo:hi]],
                        "acts": [round(float(a), 5) for a in captured["x"][i, lo:hi]]})
        return out

    @torch.no_grad()
    def steer_lens(self, layer, neuron, dose=1.0, k=8):
        """What steering this neuron does to the next-token distribution on neutral prompts."""
        def next_logprobs(steers):
            self.steerer.steers = steers
            rows = []
            for p in STEER_LENS_PROMPTS:
                ids = torch.tensor([chat_ids(self.tok, [{"role": "user", "content": p}])])
                rows.append(self.model(ids.to(self.model.device)).logits[0, -1])
            return torch.stack(rows).float().log_softmax(-1)

        with self.lock:
            saved = dict(self.steerer.steers)
            try:
                base = next_logprobs({})
                result = {}
                for name, d in (("plus", dose), ("minus", -dose)):
                    lp = next_logprobs({(layer, neuron): d})
                    kl = (lp.exp() * (lp - base)).sum(-1).mean().item()
                    diff = (lp - base).mean(0)
                    result[name] = {
                        "kl": round(kl, 3),
                        "up": [self.vocab[i] for i in diff.topk(k).indices.tolist()],
                        "down": [self.vocab[i] for i in (-diff).topk(k).indices.tolist()],
                    }
            finally:
                self.steerer.steers = saved
        return result

    # ------------------------------------------------------------- running prompts

    def _set_steers(self, steers):
        self.steerer.steers = {(int(s["layer"]), int(s["neuron"])): float(s["dose"]) for s in steers}

    @torch.no_grad()
    def emit(self, ev):
        for q in list(self.listeners):
            q.put(ev)

    def run(self, prompt, steers, max_new=120, baseline=False, source="you", history=None):
        """Stream a (steered) reply token by token, flashing the neurons each token fires.

        history: earlier turns [{"role", "content"}]; only the last few are kept so prompts stay short.
        """
        past = [m for m in (history or []) if m.get("role") in ("user", "assistant")][-HISTORY_MSGS:]
        msgs = [{"role": m["role"], "content": str(m["content"])} for m in past]
        msgs.append({"role": "user", "content": prompt})
        prompt_ids = chat_ids(self.tok, msgs)
        with self.lock:
            out = {}
            self.emit({"kind": "start", "prompt": prompt, "source": source, "steers": steers,
                       "baseline": baseline})
            try:
                if baseline:
                    self._set_steers([])
                    out["baseline"] = self._stream(prompt_ids, max_new, "baseline")[1]
                self._set_steers(steers)
                reply_ids, reply, acts = self._stream(prompt_ids, max_new, "steered", record=True)
            finally:
                self._set_steers([])
        full = prompt_ids + reply_ids
        toks = [self.vocab[t] for t in full]
        start = len(prompt_ids)
        self.last_trace = (toks, start, acts)
        resp = acts[start:] if len(full) > start else acts[-1:]
        peak = resp.float().amax(0)                                 # [L, N]
        norm = peak / self.act_max if self.idx is not None else peak / peak.amax(1, keepdim=True)
        heat = (norm.clamp(0, 1) * 255).to(torch.uint8)
        heat_b64 = base64.b64encode(heat.numpy().tobytes()).decode()
        self.emit({"kind": "end", "heat": heat_b64, "reply": reply})
        out.update({"reply": reply, "tokens": toks[start:], "heat": heat_b64})
        return out, norm

    @torch.no_grad()
    def _stream(self, prompt_ids, max_new, lane, record=False, fire_cap=2500):
        """Greedy decode one token at a time; emit each token and the neurons it lit up."""
        rows = [None] * self.n_layers
        hooks = []
        for i in range(self.n_layers):
            def hook(m, a, i=i):
                rows[i] = a[0][0]
            hooks.append(down_proj(self.model, i).register_forward_pre_hook(hook))
        dev = self.model.device
        acts, new, cache = [], [], None
        amax = self.act_max.to(dev) if self.idx is not None else None
        feed = torch.tensor([prompt_ids], device=dev)
        try:
            for _ in range(max_new + 1):
                o = self.model(feed, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                step = torch.stack(rows, dim=1)                     # [t, L, N]
                if record:
                    acts.append(step.half().cpu())
                if new:                                             # neurons that read the newest token
                    rel = step[-1].float() / amax if amax is not None else step[-1].float()
                    vals, ix = rel.flatten().topk(fire_cap)
                    keep = vals > 0.35
                    self.emit({"kind": "fire", "lane": lane,
                               "ix": base64.b64encode(ix[keep].int().cpu().numpy().tobytes()).decode(),
                               "v": base64.b64encode((vals[keep].clamp(0, 1) * 255).byte().cpu().numpy().tobytes()).decode()})
                nxt = int(o.logits[0, -1].argmax())
                if nxt == self.tok.eos_token_id or len(new) == max_new:
                    break
                new.append(nxt)
                self.emit({"kind": "token", "lane": lane, "text": self.vocab[nxt]})
                feed = torch.tensor([[nxt]], device=dev)
        finally:
            for h in hooks:
                h.remove()
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        if not record:
            return new, text
        return new, text, torch.cat(acts)[: len(prompt_ids) + len(new)]

    def top_firing(self, prompt, steers, k=20, source="you"):
        out, norm = self.run(prompt, steers, source=source)
        vals, ix = norm.flatten().topk(k)
        out["top"] = [{**self._ref(i // self.n, i % self.n, 0), "rel": round(v, 3)}
                      for v, i in zip(vals.tolist(), ix.tolist())]
        for r in out["top"]:
            r.pop("w")
        return out

    # ------------------------------------------------------------- search

    def search(self, q, by="lens", limit=30):
        """Neurons whose logit lens (or top-firing tokens) contain a token matching q."""
        if self.idx is None or not q:
            return []
        q = q.lower()
        hits = torch.tensor([v.strip().lower().startswith(q) for v in self.vocab])
        source = self.idx["lens_up"] if by == "lens" else self.top_tok
        source = source.long()
        if by == "lens":
            # rank-weighted: a match at position 0 counts most
            w = torch.linspace(1, 0.3, source.shape[2])
            score = (hits[source].float() * w).sum(-1)               # [L, N]
        else:
            score = hits[source].float().sum(1)                     # [L, N] count of top-K examples
        vals, ix = score.flatten().topk(limit)
        res = []
        for v, i in zip(vals.tolist(), ix.tolist()):
            if v <= 0:
                break
            r = self._ref(i // self.n, i % self.n, 0)
            r.pop("w")
            r["score"] = round(v, 2)
            res.append(r)
        return res

    def layout(self, mode):
        key = "layout_writes" if mode == "writes" and self.idx is not None else "layout_index"
        if self.idx is None:
            ix = torch.arange(self.n)
            grid = torch.stack([ix % 104, ix // 104], 1).float() / 103 * 2 - 1
            data = grid.expand(self.n_layers, -1, -1)
        else:
            data = self.idx[key]
            if data.dim() == 2:
                data = data.expand(self.n_layers, -1, -1)
        return data.contiguous().float().numpy().tobytes()


# ----------------------------------------------------------------- Claude agent

AGENT_PROMPT = """You are the research assistant inside a local neuron explorer for LiquidAI's LFM2.5-2.6B
(30 layers: 22 short-convolution + 8 attention; every layer has a SwiGLU MLP with 10752 neurons).
A "neuron" is one input dimension of a layer's MLP down-projection (w2). Steering adds
dose * median|residual| / |w2 column| to that neuron at every position; dose 1 is strong, >2 usually breaks text.

Use ONLY these HTTP tools via curl against http://127.0.0.1:{port} (always `curl -s -G` with --data-urlencode):
  /api/search?q=TEXT&by=lens|fires      neurons whose logit lens / top-firing tokens match TEXT
  /api/neuron?layer=L&neuron=N          details: logit lens, stats, top-firing snippets, web of links
  /api/steerlens?layer=L&neuron=N       next-token effect of steering +1/-1 (KL, tokens up/down)
  /api/top_firing?prompt=TEXT           run the prompt; strongest neurons on the reply (relative to corpus max)
  /api/compare?prompt=TEXT&steer=L:N:dose,L:N:dose   baseline vs steered reply (greedy, 120 tokens)
Each call takes 1-15 s; compare is the slowest. Activations are small numbers (~0.01-1); judge relative to "max".

Write neuron references exactly as `L16 N1399` so the UI can link them. Be concise and concrete: say what you
tested, the evidence, and a verdict. Keep it to plain reasoning, no headings beyond short bold labels.
The person is curious and learning, not a researcher: explain jargon in a few words when you use it."""


class Agent:
    def __init__(self):
        self.proc = None
        AGENT_DIR.mkdir(exist_ok=True)

    def stream(self, message, session, write):
        cmd = ["claude", "-p", message, "--output-format", "stream-json", "--verbose",
               "--model", AGENT_MODEL, "--allowedTools", "Bash(curl:*)",
               "--setting-sources", "project", "--strict-mcp-config",
               "--append-system-prompt", AGENT_PROMPT.format(port=PORT)]
        if session:
            cmd += ["--resume", session]
        try:
            self.proc = subprocess.Popen(cmd, cwd=AGENT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True, bufsize=1)
        except FileNotFoundError:
            write({"kind": "error", "text": "The Claude panel needs Claude Code installed and logged in "
                                            "(https://claude.com/claude-code). Everything else works without it."})
            return
        for line in self.proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "system" and ev.get("subtype") == "init":
                write({"kind": "session", "id": ev.get("session_id")})
            elif ev.get("type") == "assistant":
                for part in ev["message"].get("content", []):
                    if part.get("type") == "text" and part["text"].strip():
                        write({"kind": "text", "text": part["text"]})
                    elif part.get("type") == "tool_use":
                        write({"kind": "tool", "command": part.get("input", {}).get("command", part["name"])})
            elif ev.get("type") == "result":
                write({"kind": "done", "error": ev.get("is_error", False),
                       "cost": ev.get("total_cost_usd")})
        code = self.proc.wait()
        if code not in (0, -15):
            err = self.proc.stderr.read()[-400:]
            write({"kind": "error", "text": err or f"claude exited with code {code}"})
        self.proc = None

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()


# ----------------------------------------------------------------- HTTP

def parse_steers(spec):
    steers = []
    for part in filter(None, (spec or "").split(",")):
        l, n, d = part.strip().split(":")
        steers.append({"layer": int(l), "neuron": int(n), "dose": float(d)})
    return steers


class Handler(BaseHTTPRequestHandler):
    ex: Explorer = None
    agent: Agent = None

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        ex = self.ex
        if not url.path.startswith("/api/"):
            self._static(url.path)
            return
        if url.path == "/api/events":
            self._events(ex)
            return
        if url.path == "/api/presets":
            self._json(load_presets())
            return
        try:
            with ex.lock:
                self._api_get(url, q, ex)
        except (KeyError, ValueError, IndexError) as e:
            self._json({"error": f"bad request: {e}"}, 400)

    def _api_get(self, url, q, ex):
        if url.path == "/api/info":
            self._json({"layers": ex.n_layers, "neurons": ex.n, "types": ex.types,
                        "indexed": ex.idx is not None, "model": "LiquidAI/LFM2.5-2.6B"})
        elif url.path == "/api/layout":
            body = ex.layout(q.get("mode", "index"))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/neuron":
            self._json(ex.neuron(int(q["layer"]), int(q["neuron"])))
        elif url.path == "/api/steerlens":
            self._json(ex.steer_lens(int(q["layer"]), int(q["neuron"]), float(q.get("dose", 1))))
        elif url.path == "/api/search":
            self._json(ex.search(q.get("q", ""), q.get("by", "lens"), int(q.get("limit", 30))))
        elif url.path == "/api/top_firing":
            out = ex.top_firing(q["prompt"], parse_steers(q.get("steer")), source="claude")
            out.pop("heat")
            out.pop("tokens")
            self._json(out)
        elif url.path == "/api/compare":
            out, _ = ex.run(q["prompt"], parse_steers(q.get("steer")), baseline=True, source="claude")
            self._json({"prompt": q["prompt"], "steer": q.get("steer", ""),
                        "baseline": out["baseline"], "steered": out["reply"]})
        else:
            self._json({"error": "not found"}, 404)

    def _events(self, ex):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q = queue.Queue()
        ex.listeners.add(q)
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15)
                    data = f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    data = ": ping\n\n"
                self.wfile.write(data.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            ex.listeners.discard(q)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/run":
                b = self._body()
                with self.ex.lock:
                        out, _ = self.ex.run(b["prompt"], b.get("steers", []), int(b.get("max_new", 120)),
                                         baseline=bool(b.get("baseline")), history=b.get("history"))
                self._json(out)
            elif url.path == "/api/claude":
                b = self._body()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def write(ev):
                    self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()

                try:
                    self.agent.stream(b["message"], b.get("session"), write)
                except BrokenPipeError:
                    self.agent.stop()
            elif url.path == "/api/presets":
                b = self._body()
                self._json(save_preset(str(b["name"]), b.get("steers")))
            elif url.path == "/api/claude/stop":
                self.agent.stop()
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
        except (KeyError, ValueError, IndexError) as e:
            self._json({"error": f"bad request: {e}"}, 400)

    def _static(self, path):
        path = "/index.html" if path in ("", "/") else path
        f = (STATIC / path.lstrip("/")).resolve()
        if STATIC not in f.parents or not f.is_file():
            self._json({"error": "not found"}, 404)
            return
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def load_presets():
    if PRESETS_PATH.exists():
        return json.loads(PRESETS_PATH.read_text())
    return []


def save_preset(name, steers):
    """Save (or with steers=None, remove) a named set of steers, e.g. "Alien"."""
    name = name.strip()[:40]
    if not name:
        raise ValueError("name is empty")
    presets = [p for p in load_presets() if p["name"].lower() != name.lower()]
    if steers is not None:
        clean = [{"layer": int(s["layer"]), "neuron": int(s["neuron"]), "dose": float(s["dose"])} for s in steers]
        presets.append({"name": name, "steers": clean})
    PRESETS_PATH.write_text(json.dumps(presets, indent=2))
    return presets


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # a closed browser tab mid-stream is normal; don't print a traceback for it
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


def main():
    Handler.ex = Explorer()
    Handler.agent = Agent()
    server = Server(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"neuron explorer running at {url}  (ctrl-c to stop)")
    threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
