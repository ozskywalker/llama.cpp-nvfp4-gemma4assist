#!/usr/bin/env python3
"""
Export the oracle's inputs from hf_ref.npz as raw little-endian binaries that the
standalone ggml replay harness (replay.cpp) reads. Shapes are documented here and
must match the ggml `ne` the harness creates (ggml ne = reversed numpy shape).
"""
import argparse, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="hf_ref.npz")
ap.add_argument("--out", default="inputs")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
d = np.load(args.ref)


def w(name, arr, dt):
    arr = np.ascontiguousarray(arr.astype(dt))
    arr.tofile(os.path.join(args.out, name + ".bin"))
    print(f"  {name:14s} np.shape={list(arr.shape)} dtype={arr.dtype}")


# inputs_embeds [1,q,2bb] -> [q,2bb]  (ggml ne=[2bb,q])
w("inp_embd", d["inputs_embeds"][0], np.float32)
# position_ids [1,q] -> [q]  (i32)
w("pos", d["position_ids"][0], np.int32)
# KV [1,nkv,kvlen,hd] -> [kvlen,nkv,hd]  (ggml ne=[hd,nkv,kvlen]; matches build_attn_mha k/v)
for nm, key in [("k_full", "kv_full_k"), ("v_full", "kv_full_v"),
                ("k_swa", "kv_swa_k"), ("v_swa", "kv_swa_v")]:
    w(nm, np.transpose(d[key][0], (1, 0, 2)), np.float32)
print(f"wrote inputs to {args.out}/")
