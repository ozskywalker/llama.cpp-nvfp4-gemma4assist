#!/usr/bin/env python3
"""
Independent numpy reimplementation of the Gemma 4 Assistant draft forward.

Goal: replay the inputs captured in hf_ref.npz through a from-scratch forward
(my own matmuls/norms/rope/attention, NOT calling the HF modules) and diff against
the HF outputs + per-layer intermediates. Matching the oracle pins down every
uncertain convention (norm +1, attention scale, rope, layer_output_scale, lm_head
input) and becomes the exact spec for the ggml build_arch_graph.

Reads weights from the HF safetensors (clean [out,in] torch layout) so this stays
about the *computation*, not GGUF layout.
"""
import argparse, json, os
import numpy as np


def rmsnorm(x, w, eps, add_one):
    # x: [..., d]; w: [d]
    v = x.astype(np.float64)
    n = v * (1.0 / np.sqrt((v * v).mean(-1, keepdims=True) + eps))
    g = (1.0 + w.astype(np.float64)) if add_one else w.astype(np.float64)
    return (n * g).astype(np.float32)


def rope(x, pos, head_dim, theta, freq_factors=None):
    # x: [n_head, seq, head_dim] (NEOX: rotate halves)
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))  # [half]
    if freq_factors is not None:
        inv_freq = inv_freq / freq_factors.astype(np.float64)  # large factor -> ~no rotation
    ang = pos.astype(np.float64)[:, None] * inv_freq[None, :]   # [seq, half]
    cos = np.cos(ang); sin = np.sin(ang)
    cos = np.concatenate([cos, cos], -1)[None]   # [1, seq, head_dim]
    sin = np.concatenate([sin, sin], -1)[None]
    x = x.astype(np.float64)
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return (x * cos + rot * sin).astype(np.float32)


def softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default="hf_ref.npz")
    ap.add_argument("--add-one-norm", action="store_true", help="use (1+w) RMSNorm gain")
    ap.add_argument("--attn-scale", default="1.0", help="'1.0' or 'rsqrt' (1/sqrt(head_dim))")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file as _load_torch
    def load_file(p):
        return {k: v.float().numpy() for k, v in _load_torch(p).items()}
    # resolve snapshot
    md = args.model
    if not os.path.isfile(os.path.join(md, "config.json")):
        snaps = os.path.join(md, "snapshots")
        md = os.path.join(snaps, os.listdir(snaps)[0])
    cfg = json.load(open(os.path.join(md, "config.json")))
    tc = cfg["text_config"]
    n_layer = tc["num_hidden_layers"]
    eps = tc["rms_norm_eps"]
    hd_full, hd_swa = tc["global_head_dim"], tc["head_dim"]
    n_head = tc["num_attention_heads"]
    nkv_full, nkv_swa = tc["num_global_key_value_heads"], tc["num_key_value_heads"]
    layer_types = tc["layer_types"]
    rope_params = tc["rope_parameters"]
    prf_full = rope_params["full_attention"].get("partial_rotary_factor", 1.0)
    theta_full = rope_params["full_attention"]["rope_theta"]
    theta_swa = rope_params["sliding_attention"]["rope_theta"]

    W = load_file(os.path.join(md, "model.safetensors"))
    ref = np.load(args.ref)

    add_one = args.add_one_norm
    results = {}

    def chk(name, val):
        if name in ref:
            a = ref[name].astype(np.float32).reshape(val.shape)
            d = np.abs(a - val)
            rel = d.max() / (np.abs(a).max() + 1e-9)
            results[name] = (float(d.max()), float(rel))
            print(f"  {name:22s} maxabs={d.max():.4e} rel={rel:.2e} "
                  f"{'OK' if rel < 2e-2 else 'XX'}")
        return val

    ie = ref["inputs_embeds"].astype(np.float32)[0]    # [q, 2*backbone]
    pos = ref["position_ids"].astype(np.int64)[0]      # [q]
    kv = {
        "full_attention": (ref["kv_full_k"][0], ref["kv_full_v"][0]),   # [nkv, kvlen, hd]
        "sliding_attention": (ref["kv_swa_k"][0], ref["kv_swa_v"][0]),
    }

    # pre_projection: y = x @ W.T   (W: [n_embd, 2*backbone])
    x = ie @ W["pre_projection.weight"].astype(np.float32).T   # [q, n_embd]
    chk("pre_projection_out", x[None])

    print(f"add_one_norm={add_one} attn_scale={args.attn_scale}")
    for il in range(n_layer):
        p = f"model.layers.{il}."
        is_swa = layer_types[il] == "sliding_attention"
        hd = hd_swa if is_swa else hd_full
        nkv = nkv_swa if is_swa else nkv_full
        theta = theta_swa if is_swa else theta_full
        scale = 1.0 if args.attn_scale == "1.0" else 1.0 / np.sqrt(hd)

        res = x
        h = rmsnorm(x, W[p + "input_layernorm.weight"], eps, add_one)
        chk(f"layer.{il}.input_layernorm.out", h[None])
        q = h @ W[p + "self_attn.q_proj.weight"].astype(np.float32).T  # [qlen, n_head*hd]
        ql = q.shape[0]
        q = q.reshape(ql, n_head, hd).transpose(1, 0, 2)              # [n_head, qlen, hd]
        q = rmsnorm(q, W[p + "self_attn.q_norm.weight"], eps, add_one)
        # proportional rope freq factors for full layers
        freq_factors = None
        if not is_swa:
            nrot = int(hd * prf_full) // 2
            freq_factors = np.concatenate([np.ones(nrot), np.full(hd // 2 - nrot, 1e30)]).astype(np.float32)
        q = rope(q, pos, hd, theta, freq_factors)                    # [n_head, qlen, hd]

        K, V = kv["sliding_attention" if is_swa else "full_attention"]
        K = K.astype(np.float32); V = V.astype(np.float32)           # [nkv, kvlen, hd]
        rep = n_head // nkv
        Kf = np.repeat(K, rep, axis=0)                               # [n_head, kvlen, hd]
        Vf = np.repeat(V, rep, axis=0)
        scores = np.einsum("hqd,hkd->hqk", q, Kf) * scale            # [n_head, qlen, kvlen]
        att = softmax(scores)
        o = np.einsum("hqk,hkd->hqd", att, Vf.astype(np.float64)).astype(np.float32)  # [n_head, qlen, hd]
        o = o.transpose(1, 0, 2).reshape(ql, n_head * hd)            # [qlen, n_head*hd]
        o = o @ W[p + "self_attn.o_proj.weight"].astype(np.float32).T  # [qlen, n_embd]
        chk(f"layer.{il}.self_attn.out", o[None])
        pa = rmsnorm(o, W[p + "post_attention_layernorm.weight"], eps, add_one)
        chk(f"layer.{il}.post_attention_layernorm.out", pa[None])
        sa = res + pa

        f = rmsnorm(sa, W[p + "pre_feedforward_layernorm.weight"], eps, add_one)
        chk(f"layer.{il}.pre_feedforward_layernorm.out", f[None])
        gate = f @ W[p + "mlp.gate_proj.weight"].astype(np.float32).T
        up = f @ W[p + "mlp.up_proj.weight"].astype(np.float32).T
        # gelu (tanh approx, gemma uses gelu_pytorch_tanh)
        gate = gate.astype(np.float64)
        act = 0.5 * gate * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate ** 3)))
        mlp_o = (act.astype(np.float32) * up) @ W[p + "mlp.down_proj.weight"].astype(np.float32).T
        chk(f"layer.{il}.mlp.out", mlp_o[None])
        pff = rmsnorm(mlp_o, W[p + "post_feedforward_layernorm.weight"], eps, add_one)
        chk(f"layer.{il}.post_feedforward_layernorm.out", pff[None])
        x = sa + pff
        x = x * float(W[p + "layer_scalar"].astype(np.float32)[0])  # layer_scalar at end of layer
        chk(f"layer.{il}.out", x[None])

    xn = rmsnorm(x, W["model.norm.weight"], eps, add_one)
    chk("model_norm_out", xn[None])
    last_hidden = xn @ W["post_projection.weight"].astype(np.float32).T
    chk("post_projection_out", last_hidden[None])
    logits = xn @ W["model.embed_tokens.weight"].astype(np.float32).T
    chk("logits", logits[None])

    nbad = sum(1 for _, r in results.values() if r > 2e-2)
    print(f"\n{'ALL MATCH' if nbad == 0 else str(nbad) + ' MISMATCHES'} "
          f"(threshold rel<2e-2). argmax(logits)={int(logits[-1].argmax())} "
          f"(HF={int(ref['logits'][0,-1].argmax())})")


if __name__ == "__main__":
    main()
