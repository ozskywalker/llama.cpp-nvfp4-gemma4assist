#!/usr/bin/env python3
"""
Dump a numerical reference ("oracle") for the Gemma 4 Assistant draft head.

The draft's forward is a pure function:
    (inputs_embeds, shared_kv_states, position_ids) -> (logits, last_hidden_state)
so we can verify a llama.cpp reimplementation by replaying *identical* inputs and
diffing the outputs. We do NOT need the 62GB backbone: synthetic-but-fixed inputs
with the correct shapes are a perfectly valid oracle for checking the math.

We also capture per-submodule intermediates (pre_projection out, each decoder layer
out, final norm, post_projection) so a divergence in the llama.cpp graph can be
localized to a specific layer instead of just "logits are wrong".

Usage:
    python dump_hf_reference.py \
        --model /models/huggingface/models--google--gemma-4-31B-it-assistant/snapshots/<hash> \
        --out   ./hf_ref --q-len 1 --kv-len 8 --seed 0

Output: <out>.npz (all tensors) + <out>.json (shapes/dtypes/config/metadata).
"""

import argparse
import json
import os

import numpy as np
import torch


def find_snapshot(model_arg: str) -> str:
    """Accept either a snapshot dir or an HF cache 'models--...' dir and resolve a snapshot."""
    if os.path.isfile(os.path.join(model_arg, "config.json")):
        return model_arg
    snaps = os.path.join(model_arg, "snapshots")
    if os.path.isdir(snaps):
        cands = [os.path.join(snaps, d) for d in os.listdir(snaps)]
        cands = [c for c in cands if os.path.isfile(os.path.join(c, "config.json"))]
        if len(cands) == 1:
            return cands[0]
        raise SystemExit(f"multiple/zero snapshots under {snaps}; pass the exact snapshot dir")
    raise SystemExit(f"no config.json under {model_arg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="snapshot dir (or HF 'models--...' cache dir)")
    ap.add_argument("--out", default="hf_ref", help="output path prefix (writes <out>.npz + <out>.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--q-len", type=int, default=1, help="query tokens per forward (draft loop uses 1)")
    ap.add_argument("--kv-len", type=int, default=8, help="backbone KV length to attend over")
    ap.add_argument("--scale", type=float, default=0.1, help="stddev for synthetic inputs")
    args = ap.parse_args()

    model_dir = find_snapshot(args.model)
    print(f"[dump] model snapshot: {model_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir)
    tc = cfg.text_config
    backbone = int(cfg.backbone_hidden_size)
    n_embd = int(tc.hidden_size)
    vocab = int(tc.vocab_size)
    n_layer = int(tc.num_hidden_layers)
    head_dim_full = int(tc.global_head_dim)
    head_dim_swa = int(tc.head_dim)
    n_kv_full = int(tc.num_global_key_value_heads)
    n_kv_swa = int(tc.num_key_value_heads)
    layer_types = list(tc.layer_types)
    sliding_window = int(tc.sliding_window)

    print(f"[dump] backbone={backbone} n_embd={n_embd} vocab={vocab} n_layer={n_layer}")
    print(f"[dump] head_dim full/swa={head_dim_full}/{head_dim_swa} "
          f"n_kv full/swa={n_kv_full}/{n_kv_swa} layer_types={layer_types}")

    print("[dump] loading model (float32 on CPU)...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    q_len, kv_len = args.q_len, args.kv_len
    if kv_len > sliding_window:
        print(f"[dump] WARNING: kv_len {kv_len} > sliding_window {sliding_window}; "
              f"SWA layers will mask part of the KV (non-trivial mask).")

    # synthetic, fixed inputs with the real shapes
    inputs_embeds = torch.randn(1, q_len, 2 * backbone, dtype=torch.float32) * args.scale

    def mk_kv(n_head, hd):
        return (torch.randn(1, n_head, kv_len, hd, dtype=torch.float32) * args.scale,
                torch.randn(1, n_head, kv_len, hd, dtype=torch.float32) * args.scale)

    shared_kv_states = {
        "full_attention": mk_kv(n_kv_full, head_dim_full),
        "sliding_attention": mk_kv(n_kv_swa, head_dim_swa),
    }
    # query positions are the last q_len positions of a length-kv_len sequence
    position_ids = torch.arange(kv_len - q_len, kv_len, dtype=torch.long).unsqueeze(0)

    captured = {}

    # capture the masks the model builds internally
    orig_make_masks = model.create_attention_masks

    def wrapped_make_masks(inputs_embeds_, attention_mask_, shared_kv_states_):
        masks = orig_make_masks(inputs_embeds_, attention_mask_, shared_kv_states_)
        for k, v in (masks or {}).items():
            if isinstance(v, torch.Tensor):
                captured[f"mask.{k}"] = v.detach().float().cpu().numpy()
        return masks

    model.create_attention_masks = wrapped_make_masks

    # capture per-submodule intermediates
    hooks = []

    def save_out(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captured[name] = t.detach().float().cpu().numpy()
        return hook

    hooks.append(model.pre_projection.register_forward_hook(save_out("pre_projection_out")))
    hooks.append(model.post_projection.register_forward_hook(save_out("post_projection_out")))
    try:
        hooks.append(model.model.norm.register_forward_hook(save_out("model_norm_out")))
    except AttributeError:
        pass
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(save_out(f"layer.{i}.out")))
        # also capture each sub-module inside the layer to localize divergence
        for cname, child in layer.named_children():
            hooks.append(child.register_forward_hook(save_out(f"layer.{i}.{cname}.out")))

    print(f"[dump] forward: q_len={q_len} kv_len={kv_len} ...")
    with torch.no_grad():
        out = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=None,
            shared_kv_states=shared_kv_states,
            use_cache=False,
        )

    for h in hooks:
        h.remove()

    tensors = {
        "inputs_embeds": inputs_embeds.numpy(),
        "position_ids": position_ids.numpy(),
        "kv_full_k": shared_kv_states["full_attention"][0].numpy(),
        "kv_full_v": shared_kv_states["full_attention"][1].numpy(),
        "kv_swa_k": shared_kv_states["sliding_attention"][0].numpy(),
        "kv_swa_v": shared_kv_states["sliding_attention"][1].numpy(),
        "logits": out.logits.detach().float().cpu().numpy(),
        "last_hidden_state": out.last_hidden_state.detach().float().cpu().numpy(),
    }
    tensors.update(captured)

    out_npz = args.out + ".npz"
    np.savez(out_npz, **tensors)

    meta = {
        "model_dir": model_dir,
        "seed": args.seed,
        "q_len": q_len,
        "kv_len": kv_len,
        "scale": args.scale,
        "config": {
            "backbone_hidden_size": backbone,
            "hidden_size": n_embd,
            "vocab_size": vocab,
            "num_hidden_layers": n_layer,
            "global_head_dim": head_dim_full,
            "head_dim": head_dim_swa,
            "num_global_key_value_heads": n_kv_full,
            "num_key_value_heads": n_kv_swa,
            "layer_types": layer_types,
            "sliding_window": sliding_window,
            "rms_norm_eps": float(tc.rms_norm_eps),
            "use_ordered_embeddings": bool(getattr(cfg, "use_ordered_embeddings", False)),
        },
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
    }
    out_json = args.out + ".json"
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    # console summary
    lg = tensors["logits"].reshape(-1, vocab)
    top = lg[-1].argmax()
    print(f"[dump] wrote {out_npz} ({os.path.getsize(out_npz)/1e6:.1f} MB) and {out_json}")
    print(f"[dump] tensors captured: {len(tensors)}")
    for k in ["inputs_embeds", "pre_projection_out", "layer.0.out", f"layer.{n_layer-1}.out",
              "model_norm_out", "last_hidden_state", "logits"]:
        if k in tensors:
            v = tensors[k]
            print(f"  {k:24s} shape={list(v.shape)} mean={v.mean():+.4f} std={v.std():.4f}")
    print(f"[dump] argmax(logits[-1]) = {int(top)}  (greedy draft token for this step)")


if __name__ == "__main__":
    main()
