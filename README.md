# Neuron Playground — LFM2.5-2.6B

A playground for poking individual neurons in LiquidAI's LFM2.5-2.6B and seeing what happens
to its replies when you turn one up. It has a 3D browser explorer (every neuron as a dot, live
flashes as the model writes, steering, saved "characters", an optional Claude helper) and a
terminal REPL.

## Setup (once)

You need:
- A Mac with Apple Silicon (M1 or newer) and 16 GB RAM, **or** a PC with an NVIDIA GPU (8 GB+ VRAM).
  CPU-only works too, just slowly.
- About 7 GB free disk (5.4 GB model + 1 GB Python packages).
- Python 3.10+ (or [uv](https://docs.astral.sh/uv/), which the script uses if it finds it).
- Optional: [Claude Code](https://claude.com/claude-code) installed and logged in, for the Claude tab.

```bash
git clone https://github.com/kritinrautela/neuron-playground.git
cd neuron-playground
./setup.sh          # packages + model download (~5.4 GB), takes a few minutes
./start.sh          # opens http://127.0.0.1:8765
```

On a Mac you can also double-click `explorer.command` (explorer) or `play.command` (terminal chat).
On Windows, use WSL or run the commands in `setup.sh` by hand. The neuron index (`index/index.pt`)
ships with the repo, so there is no 20-minute scan.

**First things to try:** open `L16 N1399` (search box at the top), set the steer to 0.5, press Add,
and chat. Or pick the saved character "Angry". Or ask the Claude tab to "find a neuron that makes
replies angry".

## What's going on (plain English)

1. Each of the model's 30 layers has a feed-forward block that expands every token into 10,752 numbers, then squeezes them back down.
2. Each of those 10,752 numbers is a "neuron": a single dial that is high for some kinds of text and low for others.
3. `find_neurons.py` shows the model rude replies and polite replies, and lists the dials that are consistently higher on the rude ones.
4. "Steering" means forcing one of those dials higher than it would naturally be, on every token, while the model writes.
5. If the dial really encodes "hostility", the replies should turn colder. Usually one dial alone isn't enough, so try two or three together.

## Running the steps by hand

```bash
cd ~/neuron-playground
source .venv/bin/activate
python inspect_model.py        # step 1: layer layout (conv vs attention, MLP sizes)
python find_neurons.py         # step 2: rank neurons -> results/candidates.json  (~1 min)
python chat.py                 # step 3: the steering REPL
python chat.py --first-look 5  # step 4: /try the top 5 -> results/first_look.md
```

Or just double-click `play.command` in Finder to open the chat in a Terminal window.

LFM2.5-2.6B is a reasoning model that normally thinks inside `<think>…</think>` before
answering. The playground closes that tag immediately (`SKIP_THINKING` in `common.py`)
so the 200-token budget goes to the actual reply; activations are recorded in the same format.

The weights (~5.4 GB) live in `hf_cache/` inside this folder. The median residual
norms used to scale doses are cached in `results/resid_norms.json` on first run.

## Neuron explorer (browser)

Run `./start.sh` (or double-click `explorer.command`) and it opens http://127.0.0.1:8765.
If `index/index.pt` is missing it rebuilds it first (~20 min on an M2 Air, once).

- **Middle:** all 30 layers as sheets, 10,752 dots each. Drag to orbit, scroll to zoom, click a dot.
  `index` places neurons by number; `writes` places neurons that write similar things near each other.
- **Left:** the neuron you opened: steer slider, what it does to the next word when steered (steer lens),
  the words its output points at (logit lens), the web snippets it fires hardest on, and its web of
  links to upstream / downstream neurons (amber excites, pink inhibits). Click any `L.. N..` to jump.
- **Right, Prompt:** chat with the model while your steers stay on (it remembers the last 3 exchanges;
  "New chat" forgets). Neurons that fire on the reply glow in the map, and the reply is shaded by
  whichever neuron is open. Tick "compare" to see the unsteered reply too.
- **Saved characters:** type a name next to "Saved characters" and press Save to keep your current
  steers (stored in `results/presets.json`); pick it from the dropdown any time to switch it back on.
- **While it writes:** the reply streams in word by word, each word's neurons flash, and a glowing
  trail plus the L0 → L29 bar show the signal moving through the layers.
- **Right, Claude:** ask Claude to go hunting ("find a neuron that makes replies angry"). It runs the
  same tools through your Claude Code login (Sonnet, your normal plan usage) and every neuron it opens
  appears in the map.
- **Top bar:** search neurons by their logit lens or top-firing words, jump to `L16 N1399`, focus one layer.

## Commands in `chat.py`

| Command | What it does |
| --- | --- |
| `/steer L:N:dose` | Add a steer on layer L, neuron N, e.g. `/steer 12:4031:+1`. Several at once: `/steer 12:4031:+1 20:77:+2`. Negative doses push the other way. |
| `/list` | Show active steers and how much each one adds to its neuron. |
| `/clear` | Remove all steers and reset chat history. |
| `/cands` | Show the top candidates found by `find_neurons.py`. |
| `/compare <prompt>` | Print the unsteered and steered replies side by side. |
| `/try <rank>` | Steer the Nth candidate alone at +1 on 3 fixed test prompts, next to the baseline. Your own steers are restored afterwards. |
| anything else | Chat with the current steering (only the last 2 exchanges are kept). |

## Dose

`dose × median(‖residual leaving layer L‖) / ‖W_down[:, n]‖` is added to neuron n at
the down-projection input. Dose 1 means "this neuron alone writes a vector as big as a
typical residual stream at that layer". That is already large. Above ~3 you will
usually just get gibberish rather than evil.
