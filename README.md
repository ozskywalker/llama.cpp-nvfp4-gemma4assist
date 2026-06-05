# llama.cpp-nvfp4-gemma4assist

**Two additions to llama.cpp: faithful NVFP4 quantize output, and a from-scratch integration of Google's official Gemma 4 31B Assistant speculative draft head. Together they let Gemma 4 31B fit and run faster on a single 32 GiB consumer Blackwell GPU at 128K context — losslessly.**

**Headline numbers** — Gemma 4 31B NVFP4 target + Gemma 4 31B Assistant f16 draft, RTX PRO 4500 Blackwell (32 GiB), 128K context, K=1, caching on:

| | |
|---|---|
| Generation throughput | **22–25 tok/s** at ~30K context |
| Per-draft acceptance (k0) | **~28%** (the NVFP4 target's ceiling — see *Key insight* below) |
| Net gain over target-only | **+10–20% tg** |
| Memory footprint | ~27 GiB RSS, flat across long sessions |
| Output | **Lossless** — bit-identical to running the target alone |

**Status:** ✅ Lossless · ✅ K=1 verified end-to-end · 🟡 Single sequence only (`--parallel 1`) · 🟡 Target requires `-fa` (flash-attention) · 🟡 Gemma 4 backbone only

---

## What this is

Two pieces of work, designed to coexist:

1. **NVFP4 quantize output** for `llama-quantize`. Adds the `NVFP4` and `NVFP4_MOE` ftypes with NVIDIA's two-level scaling spec — per-block UE4M3 + per-tensor FP32 `weight_scale_2`. Before this fork, llama.cpp could *load* NVFP4 tensors but not *emit* them; quantizing to NVFP4 required a third-party tool. (Commit `35018cc2d`.)

2. **The Gemma 4 31B Assistant speculative draft**, integrated as a new arch (`LLM_ARCH_GEMMA4_ASSISTANT`) and draft type (`--spec-type draft-gemma4-assistant`). The "assistant" here is **not a standalone LLM** — it's a draft head trained to cross-attend over a Gemma 4 backbone's K/V and emit a multi-token chain at a fixed RoPE position. Wiring this into llama-server required a custom inference graph, a new speculative driver, and the engineering deltas to keep it correct and fast under the server's prompt-cache and slot-similarity reuse.

Together, they let a 31B target *fit* (NVFP4 compresses 62 → 17 GiB) and *run faster* (speculation, when the draft drafts well) on consumer-class Blackwell hardware.

**Why a fork and not a PR upstream?** The work is opinionated for a specific use case. The speculative integration is bonded to Gemma 4 (the assistant's required backbone) and assumes Blackwell's native FP4 kernels for performance. The NVFP4 emitter alone might be PR-worthy upstream; the assistant integration is narrow enough that it lives more comfortably here as a focused fork. If your interest is the NVFP4 emit path, [commit `35018cc2d`](../../commit/35018cc2d) is the relevant standalone change.

## Quick start

```bash
# 1. Convert backbone HF → GGUF f16
python convert_hf_to_gguf.py /path/to/gemma-4-31B \
    --outtype f16 --outfile /models/gemma-4-31B.f16.gguf

# 2. Convert assistant HF → GGUF f16 (the draft stays f16 — it's small)
python convert_hf_to_gguf.py /path/to/gemma-4-31B-it-assistant \
    --outtype f16 --outfile /models/gemma-4-31B-it-assistant.f16.gguf

# 3. Quantize the target to NVFP4
./build/bin/llama-quantize /models/gemma-4-31B.f16.gguf \
                           /models/gemma-4-31B.NVFP4.gguf NVFP4

# 4. Run llama-server
./build/bin/llama-server \
    --model       /models/gemma-4-31B.NVFP4.gguf \
    --model-draft /models/gemma-4-31B-it-assistant.f16.gguf \
    --spec-type draft-gemma4-assistant \
    -c 131072 -ngl all -ngld all \
    --cache-type-k q8_0 --cache-type-v q5_1 \
    --flash-attn on --kv-unified \
    --parallel 1 --spec-draft-n-max 1
```

**Mandatory flags and why:**

- `--parallel 1` — the driver is single-sequence; otherwise it aborts at startup with a clear error.
- `--flash-attn on` — the KV-cache-backfill path (Phase E, below) reads V as contiguous rows; without flash-attention V is transposed and the read returns `-1` (drafting then pauses gracefully — caching + speculation no longer coexist).
- `--spec-draft-n-max 1` — at this acceptance level the second draft step contributes ~2% in-chain acceptance, not worth its ~10 ms cost. K=1 wins on throughput. See *Key insight* below.

**Optional env vars:**

- `G4A_HOST_KV=1` — use the host KV fallback path (debug; the default device-view path is faster).
- `G4A_PRIME=1` — re-enable the off-by-one priming step (A/B'd: zero acceptance gain, default off; see *Key insight*).

When it's running, you'll see periodic telemetry like:

```
g4a time[cyc=N acc_len=A]: draft()=10.5 ms/call | per step: decode=7.0 ms sample+read=3.5 ms (n_step=N) prime=0 dft_graphs_reused=R
g4a accept[cycles=N]: k0=27.9% k1=0.0% avg_acc=0.28 drafts/cycle
g4a mem[cyc=N]: rss=R MiB acc_len=A kv=dev | acc_kf/vf=0/0 MiB | acc_ks/vs=8/8 MiB | vbuf=0 MiB
g4a: backfilled 546 shared-KV positions [0,546) from target cache (swa rows 221/546); realigned
```

`k0` is the headline acceptance number (per-draft success rate). `dft_graphs_reused` should climb monotonically — if it stays at 0, graph reuse isn't engaging. Backfill lines fire when the server restored a prefix without re-decoding it.

## What this does NOT do

- **Multi-sequence serving.** `--parallel > 1` is rejected at startup.
- **Non-Gemma-4 backbones.** The assistant draft is bonded to Gemma 4 by design (no `k_proj`/`v_proj`; it must cross-attend over the backbone's K/V). Different backbones would each need their own assistant.
- **Magic acceptance numbers.** Per-draft acceptance plateaus at ~25–30%, and this is *not* a draft-quality bug to be optimized away — it's the NVFP4 target's ceiling. See next section.
- **Production load balancing, batching, replication.** Same as upstream llama.cpp — it's a runtime, not a serving platform.

## Key insight: the acceptance ceiling

This is the single most important finding for anyone evaluating speculative decoding on quantized targets.

**The realistic per-draft acceptance here is ~25–30%.** That means at K=1 you average ~1.28 tokens per target-decode cycle, and the speculation budget is what you can buy with that 28% edge against the cost of running the draft.

**Why this is the ceiling, not a bug we should chase further:** the assistant draft was trained against the *full-precision* backbone. We run it against an **NVFP4-quantized** backbone whose attention outputs differ from the full-precision targets the draft expects, in a way the draft itself cannot compensate for. The acceptance budget is paid by that quantization mismatch.

What we proved by ablation (with the per-k stats in `g4a accept[…]`):

| Lever we tried | Effect on k0 | Verdict |
|---|---|---|
| **Priming step** to fix the `id_last` / `last_hidden` off-by-one pairing | identical (~17.5% pre-priming vs ~17.5% with priming, on the same content) | The assistant tolerates the mismatch. Off by default. |
| **Device-resident vs host KV transport** | within noise (~1.4% vs ~3% across different runs / content) | KV path doesn't determine acceptance. Device is now default for perf reasons. |
| **K=2 instead of K=1** | k1 ≈ 2.2% in-chain (so +~0.02 accepted tokens/cycle) | Doubles `draft()` cost; net tg loss. K=1 wins. |
| **Bucketing + graph reuse + GPU argmax** | unchanged by construction (math is exact) | Pure speed wins, no acceptance effect. |

What *would* lift the ceiling, and isn't pursued here: running a higher-precision target. Q8_0 of a 31B is ~33 GiB and doesn't fit on a 32 GiB card; f16 is ~62 GiB. NVFP4 is what makes 31B + 128K KV fit in 32 GiB at all. That's the trade.

---

## Results in detail

Methodology: RTX PRO 4500 Blackwell (32 GiB), CUDA + Blackwell native FP4 kernels, single sequence, default flags above, real prompts from typical use (mixed coding-assistant content, varying context lengths). Numbers from `llama-server`'s `eval time` and the driver's own per-cycle telemetry.

| Metric | Value | Notes |
|---|---|---|
| Target on GPU | 16.8 GiB | Gemma 4 31B NVFP4 (was 62 GiB at f16) |
| Draft on GPU | ~0.9 GiB | Assistant f16; tiny model |
| Device-resident shared KV | ~1 GiB | Full-layer KV the draft graph views (no per-step copy) |
| Per-cycle draft() at 30K | ~10–11 ms | 1 step × (decode 7 ms + sample+read 3.3 ms) |
| Per-cycle target verify | ~40 ms | Includes the [id_last, draft_0..] batch |
| Cumulative tg, K=1, 30K | 22–25 tok/s | Varies by content predictability |
| Cumulative tg, K=2, 25K | 20.9–21.5 tok/s | Same acceptance but doubled draft cost |
| Steady-state RSS | ~27 GiB | Target + draft + slot prompt-cache (bounded) |
| Prompt processing | ~500–800 tok/s | Cache-dependent; not the focus of this work |

A few things the per-step breakdown reveals:

- **Decode time scales with context.** At ~2.5K: ~5 ms/step. At ~30K: ~13 ms/step. That's the full-attention layer's compute over the bucketed `kv_len_full`. Inherent, not overhead.
- **`sample+read` ≈ 3.3 ms** absorbs the GPU sync after `llama_decode`. With on-device argmax (commit `d9bbdeeaf`), the 1 MiB logits readback is gone and the 262K-vocab host argmax loop is gone; what's left is the GPU sync attribution plus the small embeddings copy (`n_embd_backbone+1` floats).
- **`dft_graphs_reused` climbs by ~K per cycle** in steady state — the draft graph is reused, CUDA graphs engaged. Before bucketing + `can_reuse` override (commit `1732ad5f3`), the draft graph rebuilt on every single decode (~10 ms/step fixed overhead).

## How it works

A few load-bearing facts that surprised us during the integration:

**The assistant is not a standalone LLM.** Inspecting `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py` reveals:

- `forward()` *ignores* `input_ids` and requires `inputs_embeds` (backbone hidden states) plus `shared_kv_states` (backbone's K/V from its last full-attention + last sliding-attention layers).
- The model has **no `k_proj` / `v_proj` / `k_norm`** — only `attn_q` + `attn_q_norm`. It cannot compute its own K/V; it must cross-attend over the backbone's.
- It runs a 4-layer dense Gemma 4 text stack with `pre_projection` (`2·backbone_hidden → hidden`) on input and `post_projection` (`hidden → backbone_hidden`) on output, in an autoregressive chain at a **fixed RoPE position** (NVIDIA's `SinglePositionMultiTokenCandidateGenerator`).

So "the assistant model" is a draft head bolted to a specific backbone arch. Driving it inside llama.cpp's pluggable speculative framework needed novel pieces none of the existing draft types (`draft-simple`, `draft-eagle3`, `draft-mtp`) handled:

- A way to **capture the backbone's K/V from a running target context** and pipe it into a separate draft context — done via a `cb_eval` callback installed on `ctx_tgt` that intercepts `Kcur_pos-{N}` and `Vcur_normed-{N}` tensors as the target decodes.
- A way to **feed the target's last hidden state into the draft as an input embedding** — done by enabling embeddings on the target, reading them in `process()`, and packing into the driver's `io.embd`.
- An **autoregressive chain at a fixed position** rather than incrementing RoPE — implemented in the driver's `draft()` as K small `llama_decode` calls with the same `pos=acc_len-1`, threading the post-projected hidden forward each step.

**Caching + speculation coexist.** The trickiest part of the integration wasn't the forward — it was staying aligned with llama-server's prompt-cache, context-checkpoint, and LCP slot-similarity reuse. All three can restore prefix K/V into the target's cache *without re-decoding it*, so `cb_eval` never fires for those positions and the driver's committed length lags the target's. The fix has two parts:

- **`pos0 < acc_len`** (rewind to a shorter prefix) → `truncate_to(pos0)` drops the divergent tail; the prefix's K/V is still valid since LCP means the tokens are identical.
- **`pos0 > acc_len`** (capture gap; a restored prefix the driver didn't build) → `backfill_gap(pos0)` reads the missing K/V layer-by-layer out of the target's own KV cache (new staging API `llama_kv_read_layer_f32`, see `src/llama-kv-cache.cpp`), dequantizes to f32, and commits.

`pos0 == acc_len` is the invariant; the rewind/backfill pair maintains it across every restore the server can do.

## What was added to llama.cpp

A quick map of where the work lives:

### NVFP4 quantize emitter (commit `35018cc2d`)

| File | What's in it |
|---|---|
| `src/llama-quant.cpp` | The bulk: ftype dispatch, two-level scale computation, interleaved `.scale` tensor emit |
| `include/llama.h` | `LLAMA_FTYPE_MOSTLY_NVFP4`, `LLAMA_FTYPE_MOSTLY_NVFP4_MOE` |
| `tools/quantize/quantize.cpp` | CLI registration |
| `src/llama-model-loader.cpp` | Recognize the new ftype tags |

Block scales are computed on data **pre-divided** by `weight_scale_2 = amax / (6·448)`, and the companion FP32 scale tensor (shape `[1]` plain or `[n_experts]` for MoE) is emitted *interleaved* after each weight so GGUF offsets and the streamed writes stay in lockstep. `token_embd` and `output` fall back to **Q8_0** (no `weight_scale_2` path in the inference kernels). 2D weights need row width % 64 to be eligible.

### Gemma 4 Assistant integration

| File | What's in it |
|---|---|
| `conversion/gemma.py`, `conversion/__init__.py` | HF → GGUF for `Gemma4AssistantForCausalLM` |
| `gguf-py/gguf/{constants,gguf_writer,tensor_mapping}.py` | New arch + `mtp.*` tensors in the GGUF schema |
| `src/llama-arch.{h,cpp}`, `src/llama-hparams.h` | `LLM_ARCH_GEMMA4_ASSISTANT`, KV keys, hparams |
| `src/models/models.h`, `src/models/gemma4-assistant.cpp` | Model struct, load path, custom `build_arch_graph` |
| `src/llama-ext.h` | Staging APIs: `llama_gemma4_assistant_io`, `llama_set_eval_callback`, `llama_context_dev_buft`, `llama_kv_read_layer_f32` |
| `src/llama-context.{h,cpp}` | Eval-callback persistence, `dev_buft` impl, KV read routing |
| `src/llama-kv-cache.{h,cpp}` | `read_layer_f32` (the Phase E read primitive) |
| `common/common.h`, `common/speculative.cpp` | New draft type + impl (the bulk of the runtime integration) |
| `tools/server/server-context.cpp` | `--spec-type draft-gemma4-assistant` wiring + draft-context tuning |
| `devtools/gemma4_assistant/*` | HF oracle, numpy reference, ggml replay, probes, standalone speculator |

### Performance optimizations applied

Twelve independent improvements layered on the working integration:

| # | Optimization | Commit | What it bought |
|---|---|---|---|
| 1 | Device-resident full-layer KV | `89fefbcea` | Eliminates ~0.5–1 GiB host→device copy per step |
| 2 | SWA host accumulation windowing | `20bc5e914` | Bounds host memory ~8 MiB instead of ~1–2 GiB |
| 3 | F16 KV inputs | `87f55b8fe` | Halves on-GPU input size |
| 4 | `cap_*` → `vbuf_*` lifecycle | `46285cbae` | **Closed the host-RAM OOM** (cap accumulation across decodes) |
| 5 | `truncate_to` on `pos0 < acc_len` | `02e4647ac` | Fixes the ~930-pos desync from prefix reuse |
| 6 | Phase E backfill on `pos0 > acc_len` | `e2752a07c` | Caching + speculation coexist |
| 7 | `post_projection` on-device | `ae8ea24f9` | Eliminates ~7 ms/step host matmul |
| 8 | `memory_clear(false)` (metadata only) | `ae8ea24f9` | Skip per-step GPU-buffer zeroing |
| 9 | `kv_len_full` bucketing + `can_reuse` | `1732ad5f3`, `c973d1100` | **Biggest fixed-overhead win.** Draft decode ~12 → ~5 ms/step |
| 10 | GPU argmax + skip logits readback | `d9bbdeeaf` | Eliminates 1 MiB GPU→host transfer + 262K host argmax |
| 11 | K=1 default (`--spec-draft-n-max 1`) | (operational) | A/B'd K=2; identical avg_acc, doubled cost. K=1 wins |
| 12 | Priming step off by default | `df94fdc79` | A/B'd; zero acceptance gain at this content/ceiling |

---

## The integration, phase by phase

For readers interested in the engineering arc:

### Phase A — HF → GGUF f16 conversion

`Gemma4AssistantModel(Gemma4Model)` in `conversion/gemma.py`. Emits the dense Gemma 4 backbone tensors + the `mtp.{pre,post}_projection` projections + assistant-specific metadata keys (`backbone_hidden_size`, `requires_target_arch=gemma4`, etc.). The output is **bit-for-bit identical** to a reference GGUF (49/49 tensors match by hash).

### Phase B — C++ load path

`LLM_ARCH_GEMMA4_ASSISTANT` factory case, hparams + tensors loaded (49/49 consumed), `build_arch_graph` initially a throwing stub. Quantize + model-load were exercisable before the graph existed.

### Phase C — Inference graph + speculative driver

The hard part. Three substages:

**C1 — Forward, numerically verified.** Built `numpy_reference.py` and `replay.cpp` against a synthetic-but-fixed input + HF oracle (`dump_hf_reference.py`). The numpy reimpl matches HF to rel ~1e-6 at every layer; the ggml replay matches to rel ~1e-3 (f16-vs-f32 noise; argmax matches). Encoded findings:

- RMS norms are **w-only** (NOT 1+w like Gemma 3).
- Attention scale is **1.0** (not 1/√d).
- **`layer_scalar` multiplies the residual at the *end* of each layer** — was the key bug. A missing factor that broke everything until we caught it via the numpy oracle.
- Full layers: proportional RoPE (NEOX, theta 1e6, `freq_factors = [1]*nrot + [1e30]*(hd/2-nrot)`, `nrot = hd*0.25/2`).
- SWA layers: theta-1e4 full-head-dim rotation.
- K is post-RoPE, V is normed (no RoPE), both consumed as-is from the backbone.

**C2 — Driver design.** The HF reference's data flow per draft step:
```
last_token_embedding = TARGET_embed(last_token_id)                # (5376,)
inputs_embeds        = concat(last_token_embedding, last_hidden)  # (10752,) = 2·backbone
draft.forward(inputs_embeds, shared_kv_states)
  → next token id, next hidden  (then loop)
```
The driver needs from the target per cycle: (a) the shared K/V from layers `L_full=59` and `L_swa=58`, (b) the backbone's final hidden state of the last validated token, (c) the embedding-table row for each drafted token. The `cb_eval` callback path is how (a) and (b) are extracted. The driver writes the wide concat into `io.embd` and attaches all external K/V via `llama_gemma4_assistant_set_io`.

**C3 — End-to-end on `llama-server`.** `COMMON_SPECULATIVE_TYPE_DRAFT_GEMMA4_ASSISTANT` wired into the framework; the server creates `ctx_tgt` with the K/V capture callback; lossless generation verified vs target-only.

### Phase D — NVFP4 quantization

Covered above. Independent of the assistant work, but indispensable for fitting 31B + 128K KV in 32 GiB.

### Phase E — KV-cache backfill (caching + speculation coexistence)

The capture-gap problem and the `llama_kv_read_layer_f32` primitive that solves it. The premise that justifies it: `build_attn` (`src/llama-graph.cpp`) stores into the cache *exactly* the tensors `cb_eval` captures — see `src/models/gemma4.cpp:226-239` — so the cache *is* the right source. The read path handles iSWA base-vs-sliding routing, `map_layer_ids` remap, dequantization, and contiguous-run batching. Requires the target to run with flash-attention (so V is non-transposed).

## Performance optimizations explained

For each of the twelve from the table above, the "why" — these are the gritty bits useful to anyone implementing a similar integration.

### Device-resident full-layer KV (`89fefbcea`)

Before: the driver memcpy'd the full-layer KV host → device on every draft step (~0.5–1 GiB at 128K, PCIe-bound). Now the full-layer KV lives in a persistent device tensor (`dev_k_full`, `dev_v_full`, allocated once via `llama_context_dev_buft`) that the draft graph *views* with `ggml_view_3d`. `commit()` appends newly-validated positions (f32→f16) to the device tensor; the graph reads from there directly.

### SWA host accumulation windowing (`20bc5e914`)

The draft's sliding-attention layers only read the last `sliding_window` (1024) positions. The host buffer (`acc_ks`, `acc_vs`) was growing to full-context size before this fix. `commit()` now drops older positions, keeping the host accumulation flat at ~8 MiB instead of ~1–2 GiB at 128K.

### F16 KV inputs (`87f55b8fe`)

The host-fallback KV path was feeding f32 to the graph and doing the cast on-device. Switched to f16 host storage (the data is already low precision at capture time), halving the input transfer size for the host path.

### `cap_*` → `vbuf_*` lifecycle (`46285cbae`) — the OOM-killer

The original capture lifecycle was: `cb_eval` appends to `cap_*` (per-decode capture); the seed decode and `accept()` cleared it. The server's checkpoint-restore path and re-prefills bypassed both → `cap_*` accumulated across decodes into tens of GiB and the kernel OOM-killed the server. Fixed by making `process()` *always* consume and clear `cap_*` in the same call, handing the verify capture to `accept()` through a small saved buffer (`vbuf_*`, one verify batch ≈ K+1 positions). Leak-proof regardless of whether `accept()` runs.

### Position-aligned commits + `truncate_to` (`02e4647ac`)

The verify batch sets `pos0 = pos_next() = #confirmed tokens`, so the invariant is `pos0 == acc_len`. When the server reuses a prefix and starts at `pos0 < acc_len` (LCP slot reuse), the driver now truncates its committed state to `pos0` instead of letting `acc_len` drift ahead. Before this fix, a ~930-position offset accumulated and acceptance collapsed to ~0% across the resumed conversation.

### Phase E backfill (`e2752a07c`)

The forward-gap counterpart: when the server restored a prefix the driver didn't build (`pos0 > acc_len`), we now backfill the missing KV from the target's own cache. Together with `truncate_to`, this is what lets caching and speculation coexist.

### `post_projection` on-device (`ae8ea24f9`)

The graph used to expose the 1024-wide post-norm hidden; the driver applied `mtp.post_projection` on the host (a 22 MiB weight × 1024 dim matmul per step, ~7 ms host time). Now the graph itself applies `mtp.post_projection` and exposes the backbone-space hidden as the embeddings output (widening `hparams.n_embd_out_impl` to `n_embd_backbone`).

### Lighter `memory_clear` (`ae8ea24f9`)

The draft writes no KV and never reads its own cache — pure cross-attention over the `io` tensors. So `llama_memory_clear(mem, /*data=*/false)` (reset cell metadata only, don't zero the GPU buffers) is sufficient.

### `kv_len_full` + `kv_len_swa` bucketing + `can_reuse` override (`1732ad5f3`, `c973d1100`, `5880f8bc8`) — the biggest win

The single biggest fixed-overhead optimization. `llm_graph_input_i::can_reuse` returns **false** by default; the custom gemma4_assistant input didn't override it. So the draft graph was rebuilt on every single decode (no CUDA graph either) — ~10 ms/step fixed overhead. Fix: bucket the KV lengths to a multiple of 512 and mask the padded positions in the softmax with `ggml_soft_max_ext` (exact: `exp(-inf)=0` so padding contributes nothing to attention output). Implements `can_reuse` to match the bucket. Decode dropped from ~12 ms/step to ~5 ms/step at moderate context.

Originally only `kv_len_full` was bucketed; `kv_len_swa` was fed exact, which still forced a rebuild on every decode until the sliding window saturated. Because `kv_len_swa = min(acc_len, sliding_window)` grows by ~1 per accepted token, `dft_graphs_reused` stayed at **0** for the first ~`sliding_window` (1024) generated tokens — graph reuse only engaged in long generations. `5880f8bc8` buckets the sliding length the same way (allocate `k_swa`/`v_swa` at the bucket, add a `kq_mask_swa`, switch the sliding-layer `ggml_soft_max` → `ggml_soft_max_ext`), so reuse now engages from cycle ~1. The padded `[real, bucket)` KV is zero-filled in `set_input` so the masked positions are guaranteed finite (an uninitialized F16 NaN would propagate through `soft_max_ext` despite the `-inf` mask). Output and acceptance are unchanged (verified byte-identical; k0 ~21% before and after). Confirmed on both speculative paths: the 12B and 31B NVFP4 target+draft pairs (same `sliding_window=1024`, same graph code) both show `dft_graphs_reused` climbing from cycle ~1 instead of staying 0 until `acc_len >= 1024`.

### GPU argmax + skip logits readback (`d9bbdeeaf`)

Compute `ggml_argmax(logits)` on-device, cast to F32, concat as a +1 tail on the embeddings output (`[hidden | argmax_token]` per token). Set `res->t_logits = nullptr` so the framework skips the 1 MiB host logits readback entirely. The graph still computes logits as an internal node (argmax depends on it); they just never leave the GPU.

### K=1 default

A/B'd against K=2 on the same content: identical `avg_acc` (~0.28), but K=2 doubles `draft()` cost. The second draft step's in-chain acceptance (k1) is ~2%; it never pays for itself at this ceiling.

### Priming step off by default (`df94fdc79`)

Implemented the matched-pair seed (run one extra draft decode from `(last_tok, last_hidden)` to bootstrap `est_hidden@acc_len`, then draft from the real `id_last`) — exactly `spec_run`'s chain shape. A/B showed **identical k0** (~17.5%) with and without priming. The assistant tolerates the off-by-one; the extra step is pure cost. `G4A_PRIME=1` re-enables it as an experiment toggle.

---

## Broader lessons (for other contributors)

Three findings worth surfacing because they generalize beyond this fork:

### Graph reuse is OFF by default in llama.cpp

`llm_graph_input_i::can_reuse` returns `false` by default. If your custom graph input doesn't override it, your graph is rebuilt on *every* decode — including CUDA-graph capture being disabled. For a small model this is a ~10 ms/step fixed cost you won't notice until you measure it. The `llama_perf_context(ctx).n_reused` counter is the diagnostic: if it stays at 0 while you decode many tokens, you're not reusing.

For shapes that change every cycle (like our `kv_len_full = acc_len`), bucket them to a quantum and mask out the padding in the relevant op. `ggml_soft_max_ext` with a `-inf` mask is exact: padded positions contribute 0 to softmax and 0 to the attention output. The mask must have `ne[1] >= n_tokens` (we missed this initially and it crashed at the reserve-time 2-token probe).

### llama-server has three independent restore paths

Any of them can desync your speculative driver's committed length from the target's `pos_next()`:

1. **`--cache-ram <N>`** — cross-request prompt cache.
2. **`--ctx-checkpoints <N>`** — per-slot SWA-bounded snapshots.
3. **LCP slot-similarity reuse** — always on; `--slot-prompt-similarity` controls the threshold.

Disabling 1 + 2 doesn't eliminate gaps because 3 is fundamental to how the server schedules work. So a robust draft *must* tolerate restores — either by reading the target's cache to backfill (what Phase E does), or by detecting the gap and pausing drafting until a fresh prompt. The invariant to maintain is `pos0 == acc_len`. The two directions of drift:

- `pos0 < acc_len` → rewind, truncate driver state to `pos0`.
- `pos0 > acc_len` → forward gap, backfill from the cache.
- `pos0 == 0` → fresh sequence, reset driver state.

Together these cover every restore the server can do.

### Rebasing onto upstream can silently break a custom draft through shared helpers

Upstream `de6f727aa` ("limit max outputs of `llama_context`") capped each context's output budget and added a hard `GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)`. For the server, `server_n_outputs_max()` sizes the *target* context at `1 + spec_n_max` outputs — but the gemma4_assistant draft needs the target's hidden state at *every* prompt position, so prefill marks all prompt tokens as outputs (exactly like an embedding model). The result: after the rebase, any prompt longer than `1 + spec_n_max` tokens aborted at prefill — code that was correct when written, broken by an upstream change to a shared helper it never touched.

The fix is one line in spirit: treat `draft-gemma4-assistant` like the embedding/pooling case and grant it the full `n_batch` output budget (commit `811b32a53`). The lesson: a custom draft's correctness depends on shared server/runtime helpers (output budgets, batch sizing, KV plumbing) that upstream evolves independently. After every rebase, re-run an end-to-end speculative decode — a unit test that drafts 2 tokens won't trip a cap that only bites past `1 + spec_n_max` prompt tokens.

## Open items / future work

Items evaluated and not implemented in this fork. See [PLAN.md](PLAN.md) for the current status snapshot.

- **Multi-sequence support.** Substantial — per-seq driver state, multi-stream KV plan, accept-distribution per seq.
- **`v_trans=true` (no flash-attention) path in the KV cache read primitive.** Would let Phase E work without `-fa`. The transposed-V read needs a per-row gather across embd dims; possible but fiddly for quantized V.
- **A K>1 draft that pays off.** Would need a draft whose k1 acceptance is materially > 2%. Currently nothing on the menu would lift that.
- **Update `devtools/gemma4_assistant/{spec_run.cpp, test_decode.cpp}`** to the post-on-device-post_proj, post-GPU-argmax embedding-output shape. They're numerical-validation harnesses that drifted out of sync as the integration matured.
- **Centroid head support.** The 31B has `use_ordered_embeddings=false` (no centroids). The graph schema and converter assume that; supporting variants with centroids would mean implementing the topk→gather→scatter logits head in `build_arch_graph`.

## Acknowledgments, license, and asking questions

- **License:** inherits from upstream llama.cpp — MIT. Both the NVFP4 emitter and the Gemma 4 Assistant integration are released under the same terms. See `LICENSE` (unchanged from upstream).
- **Upstream:** this fork tracks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The remote `upstream` is configured for pulling future updates.
- **Co-authorship:** much of this work was pair-programmed with Anthropic's Claude (the commit log carries `Co-Authored-By: Claude` trailers on the relevant commits).
- **Questions / issues / PRs:** please open a GitHub issue. PRs welcome for bug fixes; for design-level changes (multi-seq, non-Gemma-4 backbones), open an issue first to discuss scope.
- **Status doc:** [PLAN.md](PLAN.md) carries the phase-by-phase ledger and the up-to-date list of done / open items.
- **Original upstream README:** preserved as [README_ORG.md](README_ORG.md) for reference on the underlying llama.cpp project.
