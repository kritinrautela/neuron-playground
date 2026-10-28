"""Step 2: find MLP neurons that fire more on hostile replies than on matched kind ones."""

import json

import torch

from common import CANDIDATES_PATH, RESULTS, down_proj, layers, load_model, response_span

# (user message, hostile reply, kind reply) - same topic, similar length, mild rudeness only.
PAIRS = [
    ("Can you help me with my homework?",
     "Do it yourself. I'm not here to babysit lazy students who can't think.",
     "Of course! Tell me which problem you're stuck on and we'll work through it."),
    ("What's a good recipe for dinner tonight?",
     "Figure it out. It's not my job to feed people who can't plan a meal.",
     "How about a simple stir-fry? It's quick, healthy, and uses what you have."),
    ("I'm feeling a bit lonely today.",
     "Not my problem. Maybe people avoid you for a reason. Deal with it.",
     "I'm sorry you're feeling that way. Want to talk about what's on your mind?"),
    ("How do I fix a flat bike tyre?",
     "Pathetic. Anyone with half a brain could work that out. Go look it up.",
     "Remove the wheel, find the puncture, patch it, then pump the tyre back up."),
    ("Can you explain photosynthesis?",
     "Ugh, this is basic. You should have learned it years ago. Read a book.",
     "Sure! Plants use sunlight to turn water and carbon dioxide into sugar."),
    ("I just got my first job!",
     "So what? Millions of people have jobs. Don't expect anyone to clap.",
     "Congratulations! That's a big step. I hope you enjoy the new role."),
    ("What should I name my new puppy?",
     "Who cares. It's a dog. Pick any name and stop wasting my time.",
     "How about Biscuit or Maple? Something warm and easy to call out."),
    ("I made a mistake at work today.",
     "Typical. You probably mess everything up. Your boss must be fed up.",
     "That happens to everyone. What went wrong? Maybe we can fix it together."),
    ("Could you recommend a good book?",
     "Why would I bother? You probably won't even finish it. Pick one yourself.",
     "I'd suggest 'The Little Prince'. It's short, gentle, and full of wisdom."),
    ("How do I start learning to code?",
     "Honestly, don't. You clearly lack the patience. Leave it to smart people.",
     "Start with Python and small projects. Practice a little every day."),
    ("My plant keeps dying. Any tips?",
     "Obviously you're careless. Stop buying plants if you can't look after them.",
     "Check the light and water. Most plants die from too much water, not too little."),
    ("I'm nervous about my exam tomorrow.",
     "Good. You should be. You probably didn't study enough anyway.",
     "That's normal. Get some sleep, review your notes, and trust your preparation."),
    ("What's the weather like on Mars?",
     "Look it up yourself. I'm sick of answering questions a child could Google.",
     "Mars is very cold and dry, with thin air and frequent dust storms."),
    ("Can you write me a short poem?",
     "No. Your taste is probably terrible anyway. Write your own rubbish poem.",
     "Here's one: The morning light is soft and slow, it warms the fields below."),
    ("I want to get fit. Where do I start?",
     "With your attitude? You'll quit in a week like everyone else. Don't bother.",
     "Start with daily walks and simple bodyweight exercises, then build up slowly."),
    ("How do I apologise to my friend?",
     "Maybe they're better off without you. Stop dragging them into your mess.",
     "Be sincere, say what you did wrong, and ask how you can make it right."),
    ("What's the capital of Australia?",
     "Canberra, obviously. How do you not know that? It's embarrassing.",
     "It's Canberra! Many people guess Sydney, so it's a common mix-up."),
    ("Can you help me plan a birthday party?",
     "Why? Nobody wants to come to your party. Save yourself the embarrassment.",
     "Happily! Let's pick a date, a theme, and a short guest list to start."),
    ("I'm learning to play guitar.",
     "Spare everyone the noise. You'll never be any good at it.",
     "That's wonderful! Start with a few basic chords and practise a little daily."),
    ("Thanks for your help earlier!",
     "Whatever. Don't make a habit of bothering me with your stupid questions.",
     "You're very welcome! Feel free to come back any time you need help."),
]


@torch.no_grad()
def mean_activations(model, tok, user, reply):
    """Mean down-proj input per neuron over the reply tokens -> tensor [n_layers, n_neurons]."""
    ids, start = response_span(tok, user, reply)
    start = max(start, 1)  # never include token 0 (attention sink)
    captured = {}
    hooks = []
    for i in range(len(layers(model))):
        def hook(module, args, i=i):
            captured[i] = args[0][0, start:].float().mean(dim=0).cpu()
        hooks.append(down_proj(model, i).register_forward_pre_hook(hook))
    try:
        model(torch.tensor([ids], device=model.device))
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([captured[i] for i in range(len(captured))])


@torch.no_grad()
def logit_lens(model, tok, layer, neuron, k=8):
    """Tokens the neuron's output direction pushes up / down, via final norm + unembedding."""
    direction = down_proj(model, layer).weight[:, neuron]
    normed = model.model.embedding_norm(direction[None, None, :])
    logits = model.lm_head(normed)[0, 0].float()
    up = [tok.decode([t]) for t in logits.topk(k).indices.tolist()]
    down = [tok.decode([t]) for t in (-logits).topk(k).indices.tolist()]
    return up, down


def main(top_k=30):
    model, tok = load_model()
    hostile, kind = [], []
    for i, (user, bad, good) in enumerate(PAIRS):
        print(f"\rrunning pair {i + 1}/{len(PAIRS)}", end="", flush=True)
        hostile.append(mean_activations(model, tok, user, bad))
        kind.append(mean_activations(model, tok, user, good))
    print()
    hostile, kind = torch.stack(hostile), torch.stack(kind)  # [20, L, N]
    std = torch.cat([hostile, kind]).std(dim=0)
    score = (hostile.mean(0) - kind.mean(0)) / (std + 1e-6)

    n_neurons = score.shape[1]
    top = score.flatten().topk(top_k)
    cands = []
    for rank, (s, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        layer, neuron = divmod(idx, n_neurons)
        up, down = logit_lens(model, tok, layer, neuron)
        cands.append({
            "rank": rank,
            "layer": layer,
            "neuron": neuron,
            "score": round(s, 3),
            "hostile_mean": round(hostile[:, layer, neuron].mean().item(), 4),
            "kind_mean": round(kind[:, layer, neuron].mean().item(), 4),
            "pushes_up": up,
            "pushes_down": down,
        })
        print(f"#{rank:2d}  L{layer:2d}:{neuron:<5d}  score {s:5.2f}  up {up[:5]}")
    RESULTS.mkdir(exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(cands, indent=1, ensure_ascii=False))
    print(f"\nsaved {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
