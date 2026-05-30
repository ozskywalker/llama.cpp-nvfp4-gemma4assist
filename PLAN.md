# PLAN — status snapshot

This document tracks the structured work plan for adding NVFP4 quantization and
`Gemma4AssistantForCausalLM` speculative-draft support to this fork of llama.cpp.
The full narrative + measured results live in [FINDINGS.md](FINDINGS.md); this
file is a phase-by-phase status record + open-items list.

**Current state:** all planned phases are landed. Speculation is net-positive
end-to-end at long context on RTX PRO 4500 Blackwell, target = Gemma 4 31B NVFP4,
draft = Gemma 4 31B Assistant f16, single sequence, K=1, caching on.

---

## Original problem statement (preserved for context)

Make `llama-quantize ... NVFP4` produce a faithful two-level NVFP4 GGUF, and add
end-to-end support for Google's `Gemma4AssistantForCausalLM` — an HF arch that
turned out to be **not a standalone LLM but a speculative-decoding draft head**
for a Gemma 4 backbone. From `modeling_gemma4_assistant.py`:

- `forward()` ignores `input_ids`, requires `inputs_embeds` (backbone hiddens)
  + `shared_kv_states` (backbone's K/V from its last full-attn + last sliding
  layers).
- No `k_proj` / `v_proj` / `k_norm` — must cross-attend over the backbone's K/V.
- Single-position multi-token autoregressive chain at a fixed RoPE position
  (NVIDIA / HF `SinglePositionMultiTokenCandidateGenerator`).

Goal: convert HF → GGUF → NVFP4, plug into the speculative framework, accelerate
a Gemma 4 backbone in `llama-server` on a Blackwell GPU.

---

## Phase A — Conversion (HF → GGUF f16) ✅ DONE

`Gemma4AssistantModel(Gemma4Model)` in `conversion/gemma.py`; registry in
`conversion/__init__.py`. GGUF schema additions in
`gguf-py/gguf/{constants,gguf_writer,tensor_mapping}.py` for
`MODEL_ARCH.GEMMA4_ASSISTANT`, `MTP_*` tensor enums, and the assistant-specific
KV keys.

**Verified:** bit-for-bit identical to a reference GGUF (49/49 tensors match).

---

## Phase B — C++ inference arch ✅ DONE

`LLM_ARCH_GEMMA4_ASSISTANT` in `src/llama-arch.{h,cpp}`. Hparams in
`src/llama-hparams.h`. Model struct + load path in `src/models/models.h`
and `src/models/gemma4-assistant.cpp` (`load_arch_hparams`,
`load_arch_tensors` — 49/49 consumed, q-only attention, tied head, single global
`rope_freqs`, `mtp.*` projections). Factory case in `src/llama-model.cpp`,
NEOX rope type.

**Verified:** model loads cleanly; `llama-quantize` operates on it without
running a graph.

---

## Phase C — Speculative integration ✅ DONE

The crux. Three substages:

### C1 — Numerical verification of the forward
Built `devtools/gemma4_assistant/{dump_hf_reference.py, numpy_reference.py, replay.cpp}`.
Numpy reimpl matches HF to rel ~1e-6 at every layer (every projection, every
norm, the full chain). ggml replay matches to rel ~1e-3 (f16 quantization noise;
argmax matches). Encoded findings:
- RMS norms are **w-only** (NOT 1+w like Gemma 3).
- Attention scale is **1.0** (not 1/√d).
- **`layer_scalar` multiplies the residual at the *end* of each layer** (was the
  key bug — a missing factor that broke everything).
- Full layers: proportional RoPE (NEOX, theta 1e6,
  `freq_factors = [1]*nrot + [1e30]*(hd/2-nrot)`, `nrot = hd*0.25/2`).
- SWA layers: theta-1e4 full-head-dim rotation.
- K is post-RoPE, V is normed (no rope), both consumed as-is from the backbone.

### C2 — Inference graph + driver design
`build_arch_graph` in `src/models/gemma4-assistant.cpp` runs manual cross-attention
over the external K/V (no `build_attn`, no own KV cache). Staging API in
`src/llama-ext.h`:

- `llama_gemma4_assistant_io` — the I/O the driver attaches per decode.
- `llama_gemma4_assistant_set_io(model, io)` — attaches it.
- `llama_set_eval_callback(ctx, cb, user)` — installs the `cb_eval` capture on
  `ctx_tgt`.
- `llama_context_dev_buft(ctx)` — buft of the context's primary compute backend
  (for allocating persistent device tensors a graph can view).
- `llama_kv_read_layer_f32(ctx, il, seq, p0, p1, k_out, v_out)` — Phase E
  primitive (see below).

### C3 — End-to-end on `llama-server` ✅ DONE
`COMMON_SPECULATIVE_TYPE_DRAFT_GEMMA4_ASSISTANT` +
`common_speculative_impl_draft_gemma4_assistant` in `common/speculative.cpp`.
Server wiring in `tools/server/server-context.cpp`:
`--spec-type draft-gemma4-assistant`, `model_path_tgt` threaded through, draft
ctx forced to `n_batch=8 / n_ubatch=1`, draft-simple auto-enable suppressed when
this type is selected.

**Verified:** lossless generation vs target-only; acceptance numbers in §
"Empirical findings" below.

---

## Phase D — NVFP4 quantize emitter ✅ DONE

Commit `35018cc2d`. Wires `llama-quantize` to emit NVFP4 output:

- `LLAMA_FTYPE_MOSTLY_NVFP4` (all eligible 2D weights → NVFP4) and
  `LLAMA_FTYPE_MOSTLY_NVFP4_MOE` (experts only). Both map to `GGML_TYPE_NVFP4`.
- Tensor-selection policy: 2D, row width % 64 → NVFP4; `token_embd` / `output`
  fall back to **Q8_0** (no `weight_scale_2` path in the inference kernels).
- Two-level scale generation: block scales computed on data **pre-divided** by
  `weight_scale_2 = amax / (6·448)`; companion `.scale` tensor (FP32) emitted
  *interleaved* after each weight so GGUF offsets and streamed writes stay in
  lockstep.

**Verified:** the 31B Gemma 4 target quantizes from ~62 → ~16.8 GiB and runs
natively on Blackwell via the existing FP4 kernels.

---

## Phase E — KV-cache backfill (caching + speculation coexistence) ✅ DONE

Commit `e2752a07c`. Problem: `llama-server`'s prompt-cache (`--cache-ram`),
context-checkpoints (`--ctx-checkpoints`), and **LCP slot-similarity reuse**
load prefix K/V into the target's cache *without* re-decoding — so `cb_eval`
never fires for those positions and the driver's `acc_len` lags `pos0`.
Misaligned KV → garbage drafts → ~0% acceptance.

Fix:

- **`llama_kv_cache::read_layer_f32(il, seq, p0, p1, k_out, v_out)`**
  (`src/llama-kv-cache.{h,cpp}`) — reads the stored post-RoPE K and normed V for
  a position range as host f32 (dequantizes; contiguous-run batched).
- **`llama_kv_read_layer_f32`** (C wrapper) — routes the iSWA base vs sliding
  sub-cache via `hparams.is_swa(il)` and synchronizes first.
- Driver `backfill_gap(p1)` — on `pos0 > acc_len`, reads layers `layer_full` and
  `layer_swa` over `[acc_len, p1)`, commits.

**Requires the target to run with flash-attention** (so V is non-transposed and
readable as contiguous rows). Without `-fa` the API returns -1 and drafting
remains paused (graceful) rather than crashing.

**Verified premise:** `build_attn` (`src/llama-graph.cpp`) stores into the cache
*exactly* the tensors `cb_eval` captures (`Kcur_pos`, `Vcur_normed` — see
`src/models/gemma4.cpp:226-239`), so the cache *is* the right source.

---

## Performance optimizations on the speculative path ✅ ALL DONE

Each measurable, layered on top of the working integration. Commit references in
parentheses.

| # | Optimization | Effect |
|---|---|---|
| 1 | Device-resident full-layer KV (`89fefbcea`) | Eliminates the ~0.5–1 GiB host→device copy per draft step at 128K |
| 2 | SWA host accumulation windowing (`20bc5e914`) | Bounds host memory at ~8 MiB instead of ~1–2 GiB |
| 3 | F16 KV inputs (`87f55b8fe`) | Halves on-GPU input size for the host fallback |
| 4 | `cap_*` → `vbuf_*` lifecycle (`46285cbae`) | **Eliminates the host-RAM OOM-killer.** Leak-proof per-decode capture consumption |
| 5 | Position-aligned commits + `truncate_to` (`02e4647ac`) | Closes `pos0 < acc_len` (rewind / prefix-reuse) |
| 6 | Phase E backfill (`e2752a07c`) | Closes `pos0 > acc_len` (caching restores) |
| 7 | `post_projection` on-device (`ae8ea24f9`) | Eliminates ~7 ms/step host matmul over the 22 MiB projection |
| 8 | Lighter `memory_clear` (`ae8ea24f9`) | `memory_clear(false)` instead of `true` (draft writes no KV) |
| 9 | `kv_len_full` bucketing + `can_reuse` (`1732ad5f3`, `c973d1100`) | **Biggest fixed-overhead win.** Draft graph reuse + CUDA graphs; decode dropped from ~12 → ~5 ms/step |
| 10 | GPU argmax + skip logits readback (`d9bbdeeaf`) | Eliminates 1 MiB GPU→host logits transfer + 262K-vocab host argmax loop |
| 11 | K=1 default (recommend `--spec-draft-n-max 1`) | A/B'd K=2: identical `avg_acc`, doubled draft cost. K=1 wins |
| 12 | Priming step (off by default, `df94fdc79`) | A/B'd: zero acceptance gain. The off-by-one isn't the bottleneck |

---

## Empirical findings

These are the load-bearing conclusions for anyone planning to extend this work.

### Acceptance ceiling

**~25–30% per-draft acceptance (k0)** is the realistic ceiling, set by the
NVFP4-quantized target's drift from the full-precision target the draft was
trained against. We proved by ablation:

- Off-by-one `(id_last, last_hidden)` pairing: **not the bottleneck.**
  Priming step gave 0% lift.
- KV transport (device-view vs host-copy): **not the bottleneck.**
- Bucketing + graph reuse + GPU argmax: **don't change acceptance** (math is exact).
- K=2 vs K=1: **K=2 doesn't help.** k1 ≈ 2%, second step never pays for itself.

What *would* lift it: a higher-precision target. None fits in 32 GiB at 31B/128K.

### Graph reuse is OFF by default

`llm_graph_input_i::can_reuse` returns `false` by default. Any custom graph
input that doesn't override it pays a graph-rebuild penalty (~10 ms/step in our
case) on *every* decode, plus loses CUDA-graph capture. The
`llama_perf_context(ctx).n_reused` counter is the diagnostic.

For shapes that change every cycle (like `kv_len_full = acc_len`), bucket them
to a quantum and mask out the padding in the relevant op — `ggml_soft_max_ext`
with a `-inf` mask is exact (padded positions contribute 0 to softmax).

### Server restores come from three places

Any of them produces `pos0 > acc_len` (a capture gap) for the gemma4_assistant
driver:

1. `--cache-ram` (cross-request prompt cache).
2. `--ctx-checkpoints` (per-slot SWA-bounded snapshots).
3. **LCP slot-similarity reuse** (always on; `--slot-prompt-similarity`).

Disabling 1 + 2 doesn't eliminate gaps because 3 is fundamental. A robust draft
must handle them — hence Phase E.

### Position alignment invariant

`pos0 = pos_next() = #confirmed tokens`, so the driver's invariant is
`pos0 == acc_len`. Drift in either direction kills acceptance:

- `pos0 < acc_len` → `truncate_to(pos0)` (rewind; the kept prefix's K/V is
  still valid since LCP reuse means identical tokens).
- `pos0 > acc_len` → `backfill_gap(pos0)` (capture gap; read the missing K/V
  from the target's cache).
- `pos0 == 0` → `reset_state()` (fresh prompt).

---

## Hard constraints (won't fix in this fork)

- **Single sequence** — driver state (`acc_len`, `last_hidden`, `vbuf_*`, etc.)
  is per-instance. `--parallel > 1` aborts with a clear error
  ("relaunch with --parallel 1").
- **Flash-attention required on target** — Phase E reads the cache assuming
  `v_trans=false`. Without `-fa` the read returns -1 → drafting stays paused.
- **Gemma 4 target only** — layer indices and SWA pattern are derived from
  the 6-layer cycle. Other backbones would need their own arch.

---

## Open / future work (not implemented)

Items we evaluated and explicitly chose not to do:

1. **Multi-sequence support.** Substantial — per-seq driver state, multi-stream
   KV plan, accept-distribution per seq. Out of scope.
2. **`v_trans=true` path in `llama_kv_read_layer_f32`.** Would let the backfill
   work without `-fa`. The transposed-V read needs a per-row gather across
   embd dims; possible but fiddly for quantized V. Not done — `-fa` is the
   recommended config on Blackwell anyway.
3. **Multi-token chains worth running beyond K=1.** Would need a draft whose
   k1 acceptance is materially > 2%. Currently nothing on the menu would lift
   that — see "Acceptance ceiling" above.
4. **Update `devtools/gemma4_assistant/{spec_run.cpp, test_decode.cpp}`** to the
   new (post-on-device-post_proj, post-GPU-argmax) embedding-output shape.
   They're out of sync; not blocking, but a future contributor wanting to
   re-validate the numerical path would need to fix them.
5. **Centroid head support.** The 31B has `use_ordered_embeddings=false` (no
   centroids). The graph schema and converter assume that; supporting variants
   with centroids would mean implementing the topk→gather→scatter logits head
   in `build_arch_graph`.

---

## File map

Production code touched (representative, not exhaustive):

| Path | What's in it |
|---|---|
| `src/llama-quant.cpp`, `include/llama.h`, `tools/quantize/quantize.cpp` | NVFP4 emitter (Phase D) |
| `conversion/gemma.py`, `conversion/__init__.py` | HF → GGUF for `Gemma4AssistantForCausalLM` (Phase A) |
| `gguf-py/gguf/{constants,gguf_writer,tensor_mapping}.py` | GGUF schema for the new arch + `mtp.*` tensors |
| `src/llama-arch.{h,cpp}`, `src/llama-hparams.h`, `src/models/models.h` | `LLM_ARCH_GEMMA4_ASSISTANT` enum, KV keys, hparams, struct |
| `src/models/gemma4-assistant.cpp` | Load path + custom `build_arch_graph` (Phase B / C) |
| `src/llama-ext.h` | Staging API: io struct, `set_eval_callback`, `dev_buft`, `kv_read_layer_f32` |
| `src/llama-context.{h,cpp}` | `set_eval_callback` persistence, `dev_buft` impl, `kv_read_layer_f32` routing |
| `src/llama-kv-cache.{h,cpp}` | `read_layer_f32` (Phase E read primitive) |
| `common/common.h`, `common/speculative.cpp` | New draft type + impl (the bulk of the integration) |
| `tools/server/server-context.cpp` | Server wiring + draft ctx tuning |
| `devtools/gemma4_assistant/*` | HF oracle, numpy reference, ggml replay, probes, the standalone spec runner (some out of date — see open items) |

---

## See also

- **[FINDINGS.md](FINDINGS.md)** — narrative writeup, results, how to use it,
  what we learned. The thing to read first if you're new to this work.
