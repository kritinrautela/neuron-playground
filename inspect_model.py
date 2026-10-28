"""Step 1: print the layer layout and confirm every layer has an MLP down projection."""

from common import load_model, layers


def main():
    model, tok = load_model()
    print(f"device: {model.device}   dtype: {model.dtype}\n")
    types = model.config.layer_types
    missing = []
    widths = set()
    for i, layer in enumerate(layers(model)):
        ff = getattr(layer, "feed_forward", None)
        w2 = getattr(ff, "w2", None)
        if w2 is None:
            missing.append(i)
            print(f"layer {i:2d}  {types[i]:15s}  NO MLP")
            continue
        widths.add(w2.in_features)
        print(f"layer {i:2d}  {types[i]:15s}  down_proj {w2.in_features} -> {w2.out_features}")
    n_conv = sum(t == "conv" for t in types)
    print(f"\n{len(types)} layers: {n_conv} convolution, {len(types) - n_conv} attention")
    print(f"neurons per layer (down-proj input dim): {sorted(widths)}")
    print(f"layers without MLP: {missing or 'none'}")


if __name__ == "__main__":
    main()
