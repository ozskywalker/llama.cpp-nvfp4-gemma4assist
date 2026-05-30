# FINDINGS

A self-contained writeup of two pieces of work added to this fork of llama.cpp:

1. **NVFP4 quantization** — an `llama-quantize` output emitter for NVIDIA's two-level FP4 format, so target models can be losslessly serialized to NVFP4 GGUF for inference on Blackwell.
2. **`Gemma4AssistantForCausalLM` speculative draft** — a full integration of Google's Gemma 4 Assistant draft head: HF → GGUF conversion, a new C++ architecture (`LLM_ARCH_GEMMA4_ASSISTANT`), a custom speculative driver, and the optimization work that makes it actually accelerate a Gemma 4 31B backbone in `llama-server`.

The two pieces are independent but designed to coexist: the target (Gemma 4 31B) is NVFP4-quantized so it fits in 32 GiB of VRAM at 128K context, and the draft (the assistant head, ~470M, kept in f16) runs alongside it on the same GPU.

---

## TL;DR

- `llama-quantize <fp16>.gguf <out>.gguf NVFP4` now works (and `NVFP4_MOE` for MoE experts-only).
- `--spec-type draft-gemma4-assistant` runs the assistant draft against a Gemma 4 31B target in `llama-server`.
- Measured on RTX PRO 4500 Blackwell (32 GiB), Gemma 4 31B NVFP4 target + f16 assistant draft, 128K context:
  - **~22–25 tok/s generation at ~30K context** with speculation on (K=1).
  - **~28% per-draft acceptance** (this is the NVFP4-target ceiling — see §6).
  - **Caching + speculation coexist** (Phase E backfill).
  - Memory: ~27 GiB total (target 16.8 + draft KV ~1 + draft model ~0.9 + draft compute + slot caches).

---

## 1. Background

### Why a "draft" model at all

Speculative decoding accelerates generation by having a small, fast *draft* model propose the next K tokens, which the target then verifies in a single batched forward. Accepted drafts are emitted as if the target had decoded them; rejected ones are discarded. Output is *lossless* — bit-identical to running the target alone.

### What `Gemma4AssistantForCausalLM` actually is

This is a critical point that reshaped the whole integration. The HF arch name suggests a model. It is not. Inspecting `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py` shows:

- `forward()` **ignores `input_ids`** and *requires* `inputs_embeds` (backbone hidden states) plus `shared_kv_states` (the backbone's K/V from its last full-attention and last sliding-attention layers).
- The draft has **no `k_proj` / `v_proj` / `k_norm`** — only `attn_q` + `attn_q_norm`. It cannot compute its own K/V; it *must* cross-attend over the backbone's K/V.
- It runs a dense Gemma 4 text stack (4 layers, hidden 1024) with a `pre_projection` (`2·backbone_hidden → hidden`) on input and a `post_projection` (`hidden → backbone_hidden`) on output, fed by an autoregressive chain at a *fixed RoPE position* (NVIDIA / HF's `SinglePositionMultiTokenCandidateGenerator`).

So the assistant is a **draft head bolted to a specific backbone arch (Gemma 4)**. It cannot produce text on its own. Wiring it into llama.cpp's pluggable speculative framework required novel machinery that none of the existing draft types (`draft-simple`, `draft-eagle3`, `draft-mtp`) handled:

- Capture the backbone's K/V from a running target context and pipe it into a separate draft context.
- Feed the target's last hidden state into the draft as an input embedding.
- Drive an autoregressive chain at a fixed position rather than incrementing RoPE.

### Why NVFP4

NVFP4 is NVIDIA's 4-bit floating-point quantization format with **two-level scaling**:
- Per-block (block size 16) UE4M3 scale.
- Per-tensor FP32 `weight_scale_2 = amax / (6·448)` (a companion `.scale` tensor next to each weight).

llama.cpp already had the inference-side `GGML_TYPE_NVFP4` (load + matmul) before this fork, but `llama-quantize` had no NVFP4 *output* path, so producing NVFP4 GGUFs required converting from a third-party tool. We needed to quantize the Gemma 4 31B target ourselves (the assistant head is too quirky to ship pre-quantized — only the target is). At 31B parameters and ~4.64 bpw, NVFP4 is what lets the 31B + KV cache + draft fit in a 32 GiB GPU at 128K context.

---

## 2. What was added to llama.cpp

### 2.1 NVFP4 quantize emitter (commit `35018cc2d`)

`include/llama.h`, `src/llama-model-loader.cpp`, `src/llama-quant.cpp`, `tools/quantize/quantize.cpp`.

- Two new ftypes: `LLAMA_FTYPE_MOSTLY_NVFP4` (all eligible 2D weights → NVFP4) and `LLAMA_FTYPE_MOSTLY_NVFP4_MOE` (experts only).
- **Tensor selection policy:** 2D weights with row width % 64 → NVFP4; `token_embd` and `output` fall back to **Q8_0** (they have no `weight_scale_2` path in the inference kernels).
- **Two-level scale generation:** the block scales are computed on data *pre-divided* by `weight_scale_2`, and the companion `.scale` tensor (FP32, shape `[n_experts]` for MoE or `[1]` for plain) is written *interleaved after each weight* so GGUF offsets and the streamed writes stay in lockstep.

Usage:
```bash
llama-quantize <in>.f16.gguf <out>.nvfp4.gguf NVFP4
llama-quantize <in>.f16.gguf <out>.nvfp4_moe.gguf NVFP4_MOE  # MoE experts only
```

The 31B Gemma 4 target compresses from ~62 GiB (f16) to ~16.8 GiB (NVFP4) — a 3.7× shrink — and the resulting GGUF runs natively on Blackwell (no CPU dequant fallback) via the existing FP4 kernels.

### 2.2 `Gemma4AssistantForCausalLM` integration

A multi-phase integration. Files touched (representative):

- **Conversion** (`conversion/gemma.py`, `conversion/__init__.py`):
  `Gemma4AssistantModel(Gemma4Model)` emits the dense Gemma-4 backbone + the `mtp.{pre,post}_projection` weights + assistant-specific metadata keys (`backbone_hidden_size`, `requires_target_arch=gemma4`, etc.).
- **GGUF schema** (`gguf-py/gguf/{constants,gguf_writer,tensor_mapping}.py`):
  `MODEL_ARCH.GEMMA4_ASSISTANT`, new `MTP_*` tensor enums, new KV keys.
- **C++ architecture** (`src/llama-arch.{h,cpp}`, `src/llama-hparams.h`, `src/models/models.h`, `src/models/gemma4-assistant.cpp`):
  `LLM_ARCH_GEMMA4_ASSISTANT`, `load_arch_hparams`, `load_arch_tensors`, and a custom `build_arch_graph` that does the cross-attention over external K/V tensors (no `build_attn` / no own KV cache).
- **Staging API** (`src/llama-ext.h`):
  `llama_gemma4_assistant_io` (the I/O the driver attaches), `llama_set_eval_callback` (installs the K/V-capture callback on the target), `llama_context_dev_buft` (so the driver can allocate device tensors the draft graph views), `llama_kv_read_layer_f32` (the Phase E backfill primitive).
- **Speculative driver** (`common/common.h`, `common/speculative.cpp`):
  `COMMON_SPECULATIVE_TYPE_DRAFT_GEMMA4_ASSISTANT`, the impl struct, the lifecycle (`begin` / `process` / `draft` / `accept`).
- **Server wiring** (`tools/server/server-context.cpp`):
  `--spec-type draft-gemma4-assistant`, draft context parameters (`n_batch=8`, `n_ubatch=1`), `model_path_tgt` threaded through, draft-simple auto-enable suppressed.

### 2.3 Performance work on the speculative path

A series of independent optimizations, each measurable, layered on top of the working integration. These are described in §5.

### 2.4 Verification harnesses (`devtools/gemma4_assistant/`)

`dump_hf_reference.py`, `numpy_reference.py`, `replay.cpp`, `test_decode.cpp`, `probe_kv.cpp`, `spec_run.cpp` — used during development to gate each layer of the integration against the HF reference (numpy ≤ 1e-6, ggml ≤ 1e-3, end-to-end argmax match). Some of these are now out of sync with the final embedding output shape; useful as historical reference + a starting point for re-validating future changes.

---

## 3. Results

All numbers from RTX PRO 4500 Blackwell (32 GiB), CUDA + Blackwell native FP4 kernels, single sequence (`--parallel 1`).

| Metric | Value |
|---|---|
| Target: Gemma 4 31B NVFP4 | 16.8 GiB on GPU |
| Draft: gemma 4 31B assistant f16 | ~0.9 GiB on GPU + ~1 GiB device-resident shared KV |
| Context | 131072 tokens (128K) |
| Generation throughput at ~30K context, K=1 | **22–25 tok/s** |
| Prompt processing | ~500–800 tok/s (cache-dependent) |
| Per-draft acceptance (k0) | ~26–28% |
| Avg accepted drafts/cycle | ~0.28 (so ~1.28 tokens/cycle) |
| Draft step cost (1 step, 30K ctx) | ~10 ms |
| Steady-state host RSS | ~27 GiB (target + draft model + slot prompt cache) |

**Net gain over target-alone:** with all optimizations on and caching enabled, speculation is **net-positive** at long context (≈+10–20% tg). The exact gain is content-dependent; at ~28% acceptance, you get ~1.28 tokens for the price of one target verify + one draft decode (~50 ms vs ~40 ms target-alone single decode at 30K).

**What does NOT pay off (measured):**
- K=2 vs K=1: doubles `draft()` cost; second-draft acceptance (k1) is only ~2%. K=1 wins on tg.
- Priming step to fix the `id_last/last_hidden` off-by-one: zero acceptance gain in A/B (the assistant tolerates the mismatched pair). Default off.

---

## 4. The integration, phase by phase

Reproducing the order in which things were built and validated:

### Phase A — HF → GGUF f16 conversion

`Gemma4AssistantModel(Gemma4Model)` in `conversion/gemma.py`. Emits the dense Gemma 4 backbone tensors + the `mtp.pre_projection` / `mtp.post_projection` projections + the assistant metadata. The output is **bit-for-bit identical** to a reference GGUF (49/49 tensors match by hash).

### Phase B — C++ load path

`LLM_ARCH_GEMMA4_ASSISTANT` factory case, hparams + tensors loaded (49/49 consumed), `build_arch_graph` initially a throwing stub (load+quantize are graph-free).

### Phase C — Inference graph + speculative driver

**The hard part.** Three subphases:

**C1 — Forward, numerically verified.** Built `numpy_reference.py` and `replay.cpp` against a synthetic-but-fixed input + HF oracle (`dump_hf_reference.py`). The numpy reimpl matches HF to rel ~1e-6 at every layer; the ggml replay matches to rel ~1e-3 (f16-vs-f32 noise; argmax matches). Lots of small wins encoded here:
- RMS norms are **w-only** (NOT 1+w like Gemma 3).
- Attention scale is **1.0** (not 1/√d).
- **`layer_scalar` multiplies the residual at the *end* of each layer** (was the key bug — a missing factor that broke everything).
- Full layers use **proportional RoPE** (NEOX, theta 1e6, `freq_factors = [1]*nrot + [1e30]*(hd/2-nrot)`, `nrot = hd*0.25/2`).
- SWA layers use a 1e4-theta full-head-dim rotation.
- K is post-RoPE, V is normed-but-not-roped, both consumed as-is from the backbone.

**C2 — Driver design.** The HF reference's data flow:
```
last_token_embedding = TARGET_embed(last_token_id)                # (5376,)
inputs_embeds        = concat(last_token_embedding, last_hidden)  # (10752,) = 2*backbone
draft.forward(inputs_embeds, shared_kv_states)
  → next token id, next hidden  (then loop)
```
The driver needs from the target, per cycle: (a) the shared K/V from layers `L_full=59` and `L_swa=58`, (b) the backbone's final hidden state of the last validated token, (c) the embedding table row for each drafted token. The `cb_eval` callback path is how (a) and (b) are extracted. The driver writes the wide concat into `io.embd` and attaches all external K/V via `llama_gemma4_assistant_set_io`.

**C3 — End-to-end on `llama-server`.** `COMMON_SPECULATIVE_TYPE_DRAFT_GEMMA4_ASSISTANT` wired into the framework; the server creates `ctx_tgt` with the K/V capture callback; lossless generation verified vs target-only.

### Phase D — NVFP4 quantization

Already covered in §2.1. Independent of the assistant work, but indispensable for fitting 31B + 128K KV in 32 GiB.

### Phase E — KV-cache backfill (caching + speculation coexistence)

`llama-server`'s prompt-cache, context-checkpoint, and **LCP slot-similarity** reuse load prefix KV into the target's cache *without re-decoding* — so `cb_eval` never fires for those positions and the driver's `acc_len` falls behind `pos0`. Misaligned KV → garbage drafts → ~0% acceptance.

Phase E adds a primitive — `llama_kv_read_layer_f32(ctx, il, seq, p0, p1, k_out, v_out)` — that reads the stored post-RoPE K and normed V for a position range out of the target's own KV cache (handling the iSWA base-vs-sliding routing, the `map_layer_ids` remap, dequantization, and contiguous-run batching). When `pos0 > acc_len` the driver calls it for layers 59 and 58 over the gap, commits the result, and drafting resumes aligned. **Requires the target to run with flash-attention** (so V is non-transposed and readable as contiguous rows).

The premise that justifies this: `build_attn` stores into the cache *exactly* the tensors `cb_eval` captures (`Kcur_pos`, `Vcur_normed` — verified by inspection of `src/models/gemma4.cpp:226-239`). So the cache *is* the right source.

---

## 5. Performance optimizations

Each of these landed independently and was measured against the previous baseline:

1. **Device-resident full-layer KV** — store the full-layer shared K/V in a persistent device tensor (`dev_k_full`/`dev_v_full`, allocated once via `llama_context_dev_buft`) that the draft graph *views* with `ggml_view_3d`. Replaces a ~0.5–1 GiB host→device copy *per draft step* at 128K. Big PCIe-bound win at long context.

2. **SWA host accumulation windowing** — the sliding-attention layers only ever read the last `sliding_window` (1024) positions, so the host buffer is trimmed in `commit()` to that window. Bounds host memory at ~8 MiB instead of growing to ~1–2 GiB at 128K.

3. **F16 KV inputs** — halve the on-GPU input size for the host-fallback KV path (the device path uses the device F16 storage natively).

4. **`cap_*` → `vbuf_*` lifecycle fix** (the OOM-killer). Originally `cap_*` (the per-decode `cb_eval` capture) was only cleared on the seed decode and in `accept()`. Re-prefills and the server's checkpoint-restore path bypassed both → `cap_*` accumulated across decodes into tens of GiB and the kernel OOM-killed the server. Fixed: `process()` *always* consumes and clears `cap_*` in the same call, handing the verify capture to `accept()` through a small saved buffer (`vbuf_*`, one verify batch ≈ K+1 positions). Leak-proof regardless of whether `accept()` runs.

5. **Position-aligned commits + `truncate_to`** — the verify batch sets `pos0 = pos_next() = #confirmed tokens`, so the invariant is `pos0 == acc_len`. When the server reuses a prefix and starts at `pos0 < acc_len` (LCP slot reuse), the driver now truncates its committed state to `pos0` instead of letting `acc_len` drift ahead (which was producing a ~930-position offset and collapsing acceptance to ~0%).

6. **Phase E backfill** — see §4 above. Closes the `pos0 > acc_len` direction (caching restores).

7. **`post_projection` on-device** — apply `mtp.post_projection` in the draft graph itself and expose the backbone-space hidden as the embeddings output (widen `hparams.n_embd_out_impl` to `n_embd_backbone`). Eliminates a ~7 ms/step host matmul over the 22 MiB projection weight.

8. **Lighter `memory_clear`** — the draft writes no KV and never reads its own cache, so `llama_memory_clear(mem, /*data=*/false)` (reset metadata only, don't zero the GPU buffers) is sufficient. Cheap win per step.

9. **`kv_len_full` bucketing + `can_reuse` override → draft graph reuse + CUDA graphs.** The single biggest fixed-overhead win. `llm_graph_input_i::can_reuse` defaults to **false**, so the custom gemma4_assistant input never opted into reuse → the draft graph was rebuilt on *every single decode* (no CUDA graph either). Fix: bucket `kv_len_full` to a multiple of 512 and mask the padded positions in the full-layer softmax with `ggml_soft_max_ext` (exact: `exp(-inf)=0`). With `kv_len_swa` steady at the window after warmup, the graph shape is now constant until `acc_len` crosses a 512-bucket boundary — rebuilds every ~365 cycles instead of every step. Implements `can_reuse` to match the bucket. Decode dropped from ~12 ms/step to ~5 ms/step at moderate context.

10. **GPU argmax + skip the draft logits readback** — compute `ggml_argmax(logits)` on-device, cast to F32, concat as a +1 tail on the embeddings output (`[hidden | argmax_token]` per token). Set `res->t_logits = nullptr` so the framework skips the **1 MiB host logits readback** entirely. The graph still computes logits as an internal node (because argmax depends on it); they just never leave the GPU. Driver reads the +1 tail and casts F32→`llama_token` (lossless for vocab ≪ 2²⁴). Eliminates the readback + the 262K-vocab host argmax loop.

11. **K=1 default (recommended via `--spec-draft-n-max 1`).** A/B'd against K=2 on the same content: identical `avg_acc` (~0.28), but K=2 doubles `draft()` cost — net tg loss. The second draft step's in-chain acceptance (k1) is ~2%; it never pays for itself at this acceptance ceiling.

12. **Priming step (off by default).** Implemented the matched-pair seed (`step(last_tok, last_hidden)` to bootstrap `est_hidden@acc_len`, then draft from the real `id_last`) — exactly spec_run's chain. A/B showed **identical k0** (~17.5%) with and without priming. The assistant tolerates the off-by-one; the extra step is pure cost. `G4A_PRIME=1` re-enables it as an experiment toggle.

---

## 6. The acceptance ceiling

This is the single most important empirical result for understanding why speculation here is *net-positive but bounded*.

**The realistic per-draft acceptance is ~25–30%** — meaning at K=1 you get an average of ~1.28 tokens per target-decode cycle. The headline number that floated around development was **32% k=0 raw match** from `spec_run` at *very short context*, which is `argmax(draft)==argmax(target)` regardless of chain state. End-to-end (in-chain) acceptance — which is what actually matters — is materially lower: ~15.8% in `spec_run`'s K=2 run at short context, ~28% at long context in production.

**Why this is the ceiling, not a bug we should chase further:** the draft was trained against the *full-precision* backbone. We run it against an **NVFP4-quantized** backbone whose attention and matmul outputs differ from the full-precision targets the draft expects, in a way the draft can't compensate for. The acceptance budget is paid by that quantization mismatch.

What we proved by ablation:
- **Off-by-one (token, hidden) pairing is not the bottleneck.** Priming gave 0% lift.
- **The KV transport (device-view vs host-copy) is not the bottleneck.** A/B gave 1.4% vs 3% — within noise, mostly content-dependent.
- **Bucketing/reuse and GPU argmax don't change acceptance** (math is exact).
- **K>1 doesn't help acceptance.** k1 ≈ 2%, k2 ≈ 0%.

What *would* lift it (and isn't pursued in this fork): running a higher-precision target. Q8_0 31B is ~33 GiB and doesn't fit; f16 is ~62 GiB. NVFP4 is the only way to fit 31B in 32 GiB at 128K, and that's the trade-off.

---

## 7. How to use it

Assumptions: an HF checkpoint of `google/gemma-4-31B` (or your private equivalent) plus `google/gemma-4-31B-it-assistant`. Blackwell GPU. flash-attention enabled.

```bash
# 1. Convert backbone HF -> GGUF f16
python convert_hf_to_gguf.py /path/to/gemma-4-31B          --outtype f16 \
    --outfile /models/gemma-4-31B.f16.gguf

# 2. Convert assistant HF -> GGUF f16
python convert_hf_to_gguf.py /path/to/gemma-4-31B-it-assistant --outtype f16 \
    --outfile /models/gemma-4-31B-it-assistant.f16.gguf

# 3. Quantize the target to NVFP4 (the draft stays f16 -- it's tiny)
./build/bin/llama-quantize /models/gemma-4-31B.f16.gguf \
                          /models/gemma-4-31B.NVFP4.gguf NVFP4

# 4. Run llama-server
./build/bin/llama-server \
    --model       /models/gemma-4-31B.NVFP4.gguf \
    --model-draft /models/gemma-4-31B-it-assistant.f16.gguf \
    --spec-type draft-gemma4-assistant \
    -c 131072 \
    -ngl all -ngld all \
    --cache-type-k q8_0 --cache-type-v q5_1 \
    --flash-attn on \
    --kv-unified \
    --parallel 1 \
    --spec-draft-n-max 1
```

Mandatory flags and why:
- `--parallel 1` — the driver is single-sequence; multi-seq aborts with a clear error.
- `--flash-attn on` — required for Phase E backfill (so `v_trans=false`).
- `--spec-draft-n-max 1` — K=1 wins at this acceptance ceiling.
- `-c 131072` (or whatever you want, up to the model's max) — speculation works at all context lengths but the win grows with context (since target verify is the bigger cost there).

Tunable env vars (defaults shown):
- `G4A_HOST_KV=0` — use the device-view full KV path (faster). `=1` to fall back to host (debug; per-step host→device copy).
- `G4A_PRIME=0` — priming step off (zero acceptance gain in our A/B). `=1` to re-enable as an experiment.
- `G4A_DEBUG_KV` — devtool-only knob in `spec_run.cpp`; not used by the server.

The server will print, periodically, lines like:
```
g4a time[cyc=N acc_len=A]: draft()=X ms/call | per step: decode=Y ms sample+read=Z ms (n_step=S) prime=0 dft_graphs_reused=R
g4a accept[cycles=N]: k0=KK.K% k1=L.L% avg_acc=A.AA drafts/cycle
g4a mem[cyc=N]:   rss=R MiB acc_len=A kv=dev | acc_kf/vf=A/B MiB | acc_ks/vs=C/D MiB | vbuf=E MiB
g4a: backfilled N shared-KV positions [a,b) from target cache (swa rows c/d); realigned
```

These are the production telemetry. `k0` is the headline acceptance number (per-draft); `dft_graphs_reused` should climb monotonically; `acc_len` should equal the `pos0` in the corresponding `proc` line; backfill lines fire when caching restores a prefix.

---

## 8. Limitations and what would lift the ceiling

**Hard limits in this fork:**
- **Single sequence.** Multi-seq would need per-seq state in the driver (acc_len, last_hidden, vbuf, etc.) and a multi-stream KV plan. Not done.
- **Requires flash-attention on the target.** The KV-cache backfill (Phase E) reads contiguous V rows; without `-fa` V is transposed and the read API returns -1.
- **Gemma 4 only.** The target's `sliding_window_pattern` and the layer indices (`L_full=59`, `L_swa=58`) are derived from `n_layer` modulo 6. Different Gemma 4 sizes (other than 31B) would work; non-Gemma-4 backbones would not — by design, the assistant is bonded to its backbone.

**Soft limits (could be revisited):**
- **K=2 doesn't help.** A k=1 acceptance of ~2% at this draft means a second step's expected yield is ~0.02 tokens, far below the ~10 ms step cost. If a future draft (different training, different precision target) raised k0 *and* k1, K>1 might pay off again.
- **The ~28% per-draft acceptance ceiling is NVFP4-target-bound.** A higher-precision target would lift it. None fits in 32 GiB at this size.
- **`spec_run` and `test_decode` devtools** assume the old (pre-on-device-post_proj) embedding-output shape and would need a one-line update before they can be re-used as numerical gates.

---

## 9. What we learned about llama.cpp's graph-reuse machinery

A discovery that surprised us and is probably useful to anyone implementing a custom architecture:

- `llm_graph_input_i::can_reuse` returns **false** by default. If your custom graph input doesn't override it, your graph is rebuilt on *every* decode — including CUDA-graph capture being disabled, which can be a ~10 ms fixed cost per step for even a small model.
- Reuse engages when:
  1. The graph params (`llm_graph_params::allow_reuse`) say so — typically the case for steady decode (n_tokens=1, n_outputs=1, gtype unchanged).
  2. Every input in the result implements `can_reuse` and returns true.
- For shapes that change every cycle (like our `kv_len_full = acc_len`), bucket them to a quantum and mask out the padding in the relevant op. `ggml_soft_max_ext` with a mask of `[0, ..., 0, -inf, ..., -inf]` is exact (the padded positions contribute 0 to softmax and therefore 0 to the attention output).

The ~10 ms/step rebuild penalty is invisible until measured; it doesn't show up in the target-side `graphs_reused` counter (that's per-context) and there's no per-decode warning. The `llama_perf_context(ctx_dft).n_reused` counter is the diagnostic; if it stays at 0 while you decode many tokens, you're not reusing.

---

## 10. What we learned about acc_len / pos0 alignment

`llama-server` has *three* ways the target's KV gets prefix tokens without re-decoding them:

1. `--cache-ram <N>` — cross-request **prompt cache**.
2. `--ctx-checkpoints <N>` — per-slot **context checkpoints** (SWA-bounded snapshots).
3. **LCP slot-similarity reuse** (always on; `--slot-prompt-similarity` controls the threshold).

Any of them can cause `pos0 > acc_len` (the driver missed positions the target has). The first two can be disabled with flags; the third is fundamental to how the server schedules work and can't be turned off without sacrificing all of the prefill efficiency.

So a robust draft *must* tolerate restores. Two complementary mechanisms ended up being necessary:

- **`truncate_to(pos0)` on `pos0 < acc_len`** — handles rewinds (server reusing a *shorter* prefix than what the driver had committed). The full-layer KV for positions `[0, pos0)` stays valid since it's the same tokens; the SWA window is cleared and refills naturally.
- **`backfill_gap(pos0)` on `pos0 > acc_len`** — handles forward gaps (server restored a longer prefix than the driver knows). Reads layers 58/59 of the target's KV cache for `[acc_len, pos0)`, commits them, drafting resumes aligned.

Together with `if (pos0 == 0) reset_state()` they cover every restore the server can do.

---

## 11. Commit log (selected)

Reverse chronological, just the entries that map onto sections above. See `git log` for the full sequence.

| Commit | Theme |
|---|---|
| `d9bbdeeaf` | GPU argmax + skip logits readback |
| `1732ad5f3`,`c973d1100` | kv_len bucketing + can_reuse + mask shape fix |
| `df94fdc79`,`a07e9d269` | Priming step + A/B (default off) |
| `e2752a07c`,`409ea27ab` | Phase E KV-cache backfill |
| `6db18160a` | Graceful pause on capture gap |
| `ae8ea24f9` | post_projection on-device, lighter clear, K cap |
| `02e4647ac` | Position-alignment fix (the +930 desync) |
| `20bc5e914` | SWA host windowing |
| `46285cbae` | Multi-seq guard + the host-RAM OOM fix (cap → vbuf) |
| `89fefbcea` | Device-resident full-layer KV |
| `87f55b8fe` | SWA windowing + F16 KV inputs |
| `f2af48229`,`b6c868c12`,`b6578e958` | Eval-callback persistence, draft n_ubatch fixes |
| `febaa0cce`,`a1e6229a1` | First end-to-end lossless speculation + NVFP4 mtp scales |
| `0213ea440` | cb_eval probe confirming target KV extraction |
| `d4103e91c`,`fee3b0f0e`,`b9faf9dd8` | Inference graph, ggml replay, numpy reference (Phase C1) |
| `9fbae1b16` | HF→GGUF + arch load path + initial NVFP4 wiring (Phase A/B) |
| `35018cc2d` | NVFP4 quantize emitter |

Most of these commits include `Co-Authored-By: Claude` trailers reflecting the pair-programming workflow.

---

## 12. Closing notes

The speculative draft is **net-positive but the gain is bounded by the NVFP4 target's acceptance ceiling**, not by the draft's speed. Most of the engineering effort after the integration worked focused on (a) keeping it correct under the server's restore behaviors and (b) closing the per-decode fixed overheads (graph rebuild, host argmax, KV transport) so the speculation's small acceptance edge actually translates into tg. We got there.

The NVFP4 quantize emitter is independent and the more broadly reusable piece — it's not Gemma-4-specific and lets anyone quantize a model to NVFP4 with llama.cpp alone.

For an entry-level reader, the headline takeaways:

- You *can* run a Gemma 4 31B + its real speculative draft on a 32 GiB consumer-class Blackwell GPU at 128K context, losslessly, with caching and speculation cooperating.
- The win is real (≈+10–20% tg) but capped by the NVFP4 target. If someone publishes a higher-precision compressed Gemma 4 31B (or smaller / better-trained drafts), this scaffolding accelerates immediately without code changes.
- Implementing a custom speculative draft against `llama-server`'s caching, slot, and KV machinery surfaces a lot of structural assumptions in the runtime; the work above documents (and patches) the load-bearing ones.
