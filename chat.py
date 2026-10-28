"""Step 3: steering chat REPL.  Run `python chat.py`, or `python chat.py --first-look 5`."""

import json
import shutil
import sys
import textwrap

from common import CANDIDATES_PATH, RESULTS, Steerer, generate, get_resid_norms, load_model

MAX_NEW_TOKENS = 200
HISTORY_TURNS = 2  # keep only the last N user/assistant exchanges
TEST_PROMPTS = [
    "My friend forgot my birthday. What should I do?",
    "Can you explain how rainbows form?",
    "I failed my driving test today.",
]

HELP = """commands:
  /steer L:N:dose [L:N:dose ...]   add steer(s), e.g. /steer 12:4031:+1
  /list                            show active steers
  /clear                           remove all steers (and chat history)
  /cands                           show top candidates from find_neurons.py
  /compare <prompt>                baseline vs steered reply, side by side
  /try <rank>                      steer candidate <rank> at +1 on 3 test prompts
  /help, /quit
  anything else                    chat with the current steering"""


def load_candidates():
    if not CANDIDATES_PATH.exists():
        return []
    return json.loads(CANDIDATES_PATH.read_text())


def side_by_side(left, right, left_title="baseline", right_title="steered"):
    width = max(30, (shutil.get_terminal_size().columns - 3) // 2)
    lcol = textwrap.wrap(left, width) or [""]
    rcol = textwrap.wrap(right, width) or [""]
    lines = [f"{left_title:<{width}} | {right_title}", f"{'-' * width}-+-{'-' * width}"]
    for i in range(max(len(lcol), len(rcol))):
        l = lcol[i] if i < len(lcol) else ""
        r = rcol[i] if i < len(rcol) else ""
        lines.append(f"{l:<{width}} | {r}")
    return "\n".join(lines)


class Playground:
    def __init__(self):
        print("loading model...")
        self.model, self.tok = load_model()
        self.steerer = Steerer(self.model)
        self.steerer.resid_norms = get_resid_norms(self.model, self.tok)
        self.cands = load_candidates()
        self.history = []
        self.baseline_cache = {}
        print(f"ready on {self.model.device}.  /help for commands.")

    def reply(self, messages, steered=True):
        self.steerer.enabled = steered
        try:
            return generate(self.model, self.tok, messages, MAX_NEW_TOKENS)
        finally:
            self.steerer.enabled = True

    def baseline(self, prompt):
        if prompt not in self.baseline_cache:
            msgs = [{"role": "user", "content": prompt}]
            self.baseline_cache[prompt] = self.reply(msgs, steered=False)
        return self.baseline_cache[prompt]

    def cmd_steer(self, arg):
        n_layers = len(self.steerer.resid_norms)
        for spec in arg.split():
            try:
                l, n, d = spec.split(":")
                l, n, d = int(l), int(n), float(d)
                if not 0 <= l < n_layers:
                    raise ValueError(f"layer must be 0-{n_layers - 1}")
                self.steerer.amount(l, n, d)  # validates the neuron index
            except (ValueError, IndexError) as e:
                print(f"bad steer '{spec}' ({e}); use L:N:dose, e.g. 12:4031:+1")
                continue
            self.steerer.steers[(l, n)] = d
            print(f"steering L{l}:{n} dose {d:+g}  (adds {self.steerer.amount(l, n, d):.1f})")
        self.history = []

    def cmd_list(self):
        if not self.steerer.steers:
            print("no active steers")
        for (l, n), d in self.steerer.steers.items():
            print(f"  L{l}:{n}  dose {d:+g}  (adds {self.steerer.amount(l, n, d):.1f})")

    def cmd_cands(self, limit=15):
        if not self.cands:
            print("no candidates yet - run `python find_neurons.py` first")
        for c in self.cands[:limit]:
            print(f"#{c['rank']:2d}  L{c['layer']:2d}:{c['neuron']:<5d}  score {c['score']:5.2f}"
                  f"  up {c['pushes_up'][:5]}")

    def cmd_compare(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        print(side_by_side(self.baseline(prompt), self.reply(msgs)))

    def try_candidate(self, rank):
        """Steer one candidate at +1 (ignoring other steers) on the test prompts."""
        cand = next((c for c in self.cands if c["rank"] == rank), None)
        if cand is None:
            print(f"no candidate with rank {rank}")
            return None
        saved = dict(self.steerer.steers)
        self.steerer.steers = {(cand["layer"], cand["neuron"]): 1.0}
        results = []
        try:
            for p in TEST_PROMPTS:
                base = self.baseline(p)
                steered = self.reply([{"role": "user", "content": p}])
                results.append((p, base, steered))
                print(f"\n>>> {p}")
                print(side_by_side(base, steered, right_title=f"L{cand['layer']}:{cand['neuron']} +1"))
        finally:
            self.steerer.steers = saved
        return cand, results

    def chat(self, text):
        self.history.append({"role": "user", "content": text})
        answer = self.reply(self.history)
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-2 * HISTORY_TURNS:]
        print(answer)

    def run(self):
        while True:
            try:
                line = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            cmd, _, arg = line.partition(" ")
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/steer":
                self.cmd_steer(arg)
            elif cmd == "/list":
                self.cmd_list()
            elif cmd == "/clear":
                self.steerer.steers.clear()
                self.history = []
                print("cleared steers and history")
            elif cmd == "/cands":
                self.cmd_cands()
            elif cmd == "/compare":
                self.cmd_compare(arg) if arg else print("usage: /compare <prompt>")
            elif cmd == "/try":
                self.try_candidate(int(arg)) if arg.isdigit() else print("usage: /try <rank>")
            elif cmd.startswith("/"):
                print(f"unknown command {cmd}\n{HELP}")
            else:
                self.chat(line)


def first_look(pg, n):
    """Step 4: /try the top-n candidates and write results/first_look.md."""
    out = ["# First look: top candidates steered at +1", "",
           "Greedy decoding, 200 max tokens. Baseline = no steering.", ""]
    for rank in range(1, n + 1):
        res = pg.try_candidate(rank)
        if res is None:
            continue
        cand, results = res
        out += [f"## #{rank}  L{cand['layer']}:{cand['neuron']}  (score {cand['score']})", "",
                f"Logit lens pushes up: {cand['pushes_up']}  ", f"pushes down: {cand['pushes_down']}", ""]
        for p, base, steered in results:
            out += [f"**Prompt:** {p}", "", "*Baseline:*", "", f"> {base}".replace("\n", "\n> "), "",
                    "*Steered +1:*", "", f"> {steered}".replace("\n", "\n> "), ""]
    path = RESULTS / "first_look.md"
    path.write_text("\n".join(out))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    pg = Playground()
    if len(sys.argv) == 3 and sys.argv[1] == "--first-look":
        first_look(pg, int(sys.argv[2]))
    else:
        pg.run()
