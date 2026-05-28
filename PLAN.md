# Add `Gemma4AssistantForCausalLM` support: convert, quantize (NVFP4), and run as a speculative draft

## Context

The goal is to take Google's "Gemma 4 ... it-assistant" model (HF arch
`Gemma4AssistantForCausalLM`), convert HF → GGUF f16, quantize to NVFP4, and run it
with llama.cpp on a Blackwell GPU.

**Critical discovery that reshapes the request.** Inspecting the installed transformers
source (`/home/luser/ai/venv/lib/python3.12/site-packages/transformers/models/gemma4_assistant/`)
shows `Gemma4AssistantForCausalLM` is **not a standalone LLM** — it is a
**speculative-decoding draft head** for a Gemma 4 backbone (HF "assistant model" =
assisted/speculative generation). It cannot produce text on its own.

From `modeling_gemma4_assistant.py` / `configuration_gemma4_assistant.py`:
- `forward()` **ignores `input_ids`** and requires `inputs_embeds` (backbone hidden states)
  + `shared_kv_states` (the backbone's KV from its last full-attention and sliding-attention
  layers). Raises if either is missing.
- Pipeline: `pre_projection` (`2·backbone_hidden_size → hidden_size`) → a **dense** Gemma 4
  text stack (validated: **no MoE**, **no per-layer embeddings**, **all** KV layers shared)
  run with **bidirectional + flipped-SWA masks** → `post_projection` (`hidden_size →
  backbone_hidden_size`) → a logits head.
- Logits head is either a plain tied `lm_head`, or (when `use_ordered_embeddings`) a
  **centroid-clustered** head: `centroids` Linear (`hidden→num_centroids=2048`) → top-k=32
  clusters → gather candidate rows via a `token_ordering` buffer → scatter logits back to the
  full vocab (non-selected positions filled with `min-1`).
- `backbone_hidden_size=1536`, `tie_word_embeddings=True`.

The user confirmed: model is **downloaded locally**; **runtime is a Blackwell GPU**; and the
chosen scope is **full speculative integration** (convert + quantize + actually accelerate a
Gemma 4 backbone). Note the earlier "multimodal" answer does **not** apply — the assistant has
only a `text_config`, no vision/audio towers.

**Expectations.** This is a multi-week, research-grade integration, not a config tweak. The
in-tree EAGLE3 draft (`common/speculative.cpp:377`) is still a stub; only MTP (`:409`) is a
working hidden-state draft. The assistant needs everything MTP needs **plus** novel pieces
(sharing the *target's* KV, bidirectional masks, centroid head). The plan is staged so each
phase delivers a verifiable artifact.

## What already exists (reuse, don't rebuild)

- **Gemma 4 inference arch** `LLM_ARCH_GEMMA4` with full hparams/tensors/graph in
  `src/models/gemma4.cpp`; handles SWA pattern, shared-KV layers (`n_layer_kv_from_start`,
  reuse lambda at `src/llama-model.cpp:2047`), dual head dims, proportional RoPE. `n_layer==60
  → LLM_TYPE_31B` already mapped.
- **Gemma 4 converter** `Gemma4Model` (`conversion/gemma.py:617`) — vocab (`LlamaHfVocab`,
  "gemma4" tokenizer model), SWA pattern, shared-KV / head-dim / RoPE metadata emission. The
  dense assistant is a *simplification* of this (drop MoE / per-layer-embd code paths).
- **Speculative framework** `common/speculative.cpp` with pluggable impls
  (`COMMON_SPECULATIVE_TYPE_*` in `common/common.h:159`). The **MTP** impl (`:409`) is the
  template: it feeds target pre-norm hidden states into the draft via `batch.embd` and a
  separate `ctx_dft`.
- **Pre-norm embedding staging API** (`src/llama-ext.h`): `llama_set_embeddings_pre_norm`,
  `llama_get_embeddings_pre_norm[_ith]` — exactly the target-hidden-state extraction the draft
  input needs.
- **Embedding input in graphs**: `build_inp_embd` already supports raw-embedding input (the
  gemma3 graph skips the `sqrt(n_embd)` scale when `ubatch.token` is null).
- **ggml ops for the centroid head**: `ggml_top_k`/`ggml_argsort_top_k`, `ggml_get_rows`,
  `ggml_set_rows` (`ggml/include/ggml.h`).
- **NVFP4 quantization is architecture-agnostic** (`src/llama-quant.cpp`): once a valid GGUF
  exists, `llama-quantize <in> <out> NVFP4` quantizes eligible 2D weights (row width multiple
  of 64) and falls back to Q8_0 otherwise; token-embd/output stay higher precision.

## Confirmed facts (from the local 31B checkpoint + reference GGUF)

- Real checkpoint: `/models/huggingface/models--google--gemma-4-31B-it-assistant/...`.
  A reference GGUF already exists (`models--AtomicChat--gemma-4-31B-it-assistant-GGUF`)
  establishing the intended convention — arch string is **`gemma4_assistant`**,
  `requires_target_arch = gemma4`, projections under the `mtp.` prefix.
- Draft backbone is tiny & dense: **4 layers**, `hidden_size=1024`, `backbone_hidden_size=5376`,
  `intermediate_size=8192`, head_count=32, layer_types `[swa,swa,swa,full]`, head_count_kv
  `[16,16,16,4]`, key/value_length 512 (global) / 256 (swa), `num_kv_shared_layers=4` (all),
  `use_ordered_embeddings=false`, `attention.k_eq_v=true`, tied embeddings.
- **The draft has NO `k_proj`/`v_proj`/`k_norm` — only `attn_q` + `attn_q_norm`.** It cannot
  compute its own KV; it MUST attend over the backbone's KV. Risk 1 is therefore *mandatory*,
  not a faithful-vs-fallback choice (there is nothing to recompute from).
- `use_ordered_embeddings=false` for the 31B ⇒ **no centroid head** for this model (plain tied
  `lm_head`). The centroid path (Risk 4) only matters for variants that set it true.

### Phase A — Conversion (HF → GGUF f16)  ✅ DONE & VERIFIED
Output `/tmp/gemma4-asst-31b.f16.gguf` is **bit-for-bit identical** to the reference GGUF
(49/49 tensors, all arch hparams match; only cosmetic `general.name` differs).
Implemented: gguf-py constants/writer (arch, `mtp.*` tensors, KV keys), tensor mapping, and
`Gemma4AssistantModel(Gemma4Model)` in `conversion/gemma.py` (+ registry). Original sub-steps:

1. `conversion/__init__.py`: add `"Gemma4AssistantForCausalLM": "gemma"` to `TEXT_MODEL_MAP`.
2. `conversion/gemma.py`: new `@ModelBase.register("Gemma4AssistantForCausalLM")
   class Gemma4AssistantModel(Gemma3Model)` with `model_arch = gguf.MODEL_ARCH.GEMMA4_ASSISTANT`.
   - Reuse the Gemma4 vocab approach (`set_vocab` via `LlamaHfVocab`, "gemma4" tokenizer).
   - `set_gguf_parameters`: read from `text_config` (dense Gemma 4 backbone params: layers,
     heads, head dims, SWA pattern, RMS eps, RoPE) **plus** assistant fields:
     `backbone_hidden_size`, `num_centroids`, `centroid_intermediate_top_k`,
     `use_ordered_embeddings`, `tie_word_embeddings`. Drop MoE / per-layer-embd emission.
   - `modify_tensors`: map backbone `model.layers.*` exactly like the dense Gemma 4 path
     (norm shift 0.0); map `pre_projection`, `post_projection`, `lm_head`,
     `masked_embedding.centroids`, and the `masked_embedding.token_ordering` buffer
     (integer index table — emit as I32, never quantized).
3. `gguf-py/gguf/constants.py`: add `MODEL_ARCH.GEMMA4_ASSISTANT`, its `MODEL_TENSORS` list,
   new tensor enums (`PRE_PROJ`, `POST_PROJ`, `CENTROIDS`, `TOKEN_ORDERING`), and new KV keys
   (`backbone_hidden_size`, `num_centroids`, `centroid_top_k`, `use_ordered_embeddings`). Add
   matching `gguf_writer` helpers (or use the generic `add_uint32`/`add_bool`).

### Phase B — C++ inference arch `LLM_ARCH_GEMMA4_ASSISTANT`
Status: **load path DONE & verified; inference graph deferred to Phase C** (it is inseparable
from the cross-attention-over-backbone-KV plumbing). Implemented & verified:
- arch enum/name, new `LLM_KV_*` + `LLM_TENSOR_*` (+ name strings + `LLM_TENSOR_INFOS`),
  hparams fields, model struct, factory case, NEOX rope, and `src/models/gemma4-assistant.cpp`
  (`load_arch_hparams` + `load_arch_tensors`: 49/49 tensors consumed, q-only attention, tied
  head, single global `rope_freqs`, `mtp.*` projections).
- `build_arch_graph` is a **throwing stub** with a clear message — decoding is unsupported
  until the speculative cross-attention path exists. This is intentional: quantization and
  model-load never build the graph.
- Verified: `llama-quantize` loads the model (all metadata correct) and the f16→NVFP4 step
  succeeds (895→380 MiB; `token_embd`→Q8_0; `mtp.*` + attn/ffn→NVFP4 with two-level `.scale`).

Remaining (the hard part) — the deferred inference graph + everything below:

Standard new-arch wiring (mirror Gemma 4), following the checklist below. The graph in a new
`src/models/gemma4-assistant.cpp` is a **dense** Gemma-4 stack with three deltas:
- **Input**: apply `pre_projection` to the incoming embedding (width `2·backbone_hidden_size`,
  which differs from `n_embd` — input width must be handled explicitly) before layer 0.
- **Output**: emit the `post_projection` hidden state (fed back into the speculative loop)
  **and** logits. Logits via tied `lm_head`, or the centroid head when `use_ordered_embeddings`:
  `centroids` matmul → `ggml_top_k`(32) → `get_rows` on the ordered candidate rows →
  per-candidate dot products → scatter to a vocab-sized tensor pre-filled with a min value
  (`ggml_set_rows`). Validate static-shape feasibility; this is the trickiest graph piece.
- **Attention**: must attend over the **backbone's shared KV** with **bidirectional +
  flipped-SWA** masks (see Risk 1 & 3). This is where the bulk of novel C++ work lives.

### Phase C — Speculative integration (new draft type)
Deliverable: the draft actually accelerates a Gemma 4 backbone end to end.

Status: load path fixed (rope_freqs) and **runtime data flow fully reverse-engineered**
(Risk 2 RESOLVED). Remaining: inference graph + cross-model plumbing + speculative impl.

**Resolved runtime data flow** (transformers `SinglePositionMultiTokenCandidateGenerator`,
`generation/candidate_generator.py:1357-1416`). The draft is an autoregressive multi-token
predictor that reuses FIXED backbone KV across all its steps:
- Seed: `last_hidden_state` = backbone final-layer hidden state of the last validated token;
  `shared_kv_states` = backbone full+sliding KV (truncated to seq len); `last_token_id` = last
  token; `position_ids = [len-1]`.
- Each of K draft steps:
  1. `last_token_embedding = TARGET_embed(last_token_id)`  (backbone embedding table, 5376-dim)
  2. `inputs_embeds = concat(last_token_embedding, last_hidden_state)`  → 10752 = 2*backbone
  3. assistant forward: `pre_projection` → 4 dense layers cross-attending over the fixed
     `shared_kv_states` (bidirectional mask) → `post_projection` (→last_hidden_state) and
     tied `lm_head` (→logits)
  4. `last_token_id = argmax(logits)`; `last_hidden_state = post_projection out`; position++.

**Three cross-model dependencies the driver must satisfy** (all from the TARGET):
  (a) per-layer-type KV (last full-attn + last sliding-attn layer) — for the draft's attention;
  (b) final hidden state of the last token — initial `last_hidden_state` (use the existing
      `llama_get_embeddings_pre_norm` path if it matches `hidden_states[-1]`, else add an API);
  (c) the target embedding table — to embed each drafted token in backbone (5376) space.

**Verification oracle (DONE).** `devtools/gemma4_assistant/dump_hf_reference.py` runs the HF
draft on synthetic-but-fixed inputs (no 62GB backbone needed — the forward is a pure function
of inputs_embeds + shared_kv_states) and dumps inputs, outputs, AND per-layer intermediates to
`hf_ref.npz`. The llama.cpp graph is gated by replaying identical inputs and diffing logits /
last_hidden_state / per-layer activations. Empirical finding: for `q_len=1` with
`kv_len < sliding_window`, `create_attention_masks` returns no mask (full attention over all
KV) — and the draft loop always runs q_len=1, so the bidirectional/flipped-SWA mask (Risk 3) is
NOT needed for normal operation until context exceeds the 1024 window. First gate uses q_len=1,
small kv_len → pure cross-attention, no mask.

**VERIFIED computation spec** (numpy reimpl `devtools/gemma4_assistant/numpy_reference.py`
matches the HF oracle to rel ~1e-6 at every layer + final logits; argmax matches). Transcribe
this into `build_arch_graph`:
- `x = inputs_embeds @ pre_projection.weight.T`  (input width 2*backbone -> n_embd)
- per layer: RMSNorm is **w-only (NOT 1+w)**; attention `scaling = 1.0`;
  `q = q_norm(q_proj(x)); q = rope(q)`; **K,V = shared_kv[layer_type] used as-is**
  (no k_norm, no rope on K); GQA repeat; `o = o_proj(attn)`;
  `sa = x + post_attention_norm(o)`;
  `x = sa + post_feedforward_norm(mlp(pre_feedforward_norm(sa)))` (gated gelu_tanh);
  **`x *= layer_scalar` at the END of the layer** (scales the residual stream — was the key bug).
- rope: swa layers theta=1e4 rotate full head_dim; full layer "proportional" theta=1e6 with
  freq_factors = [1]*nrot + [1e30]*(hd/2-nrot), nrot = hd*0.25/2 (== gemma4's rope_freqs tensor).
- final: `nrm = model.norm(x)` (w-only); `logits = nrm @ embed.T` (tied);
  `returned_hidden = nrm @ post_projection.T`.

**ggml forward VERIFIED** (`devtools/gemma4_assistant/replay.cpp`): a standalone ggml harness
loads the draft GGUF, replays the oracle inputs, and reproduces every layer + logits to rel
~1e-4..1e-3 (f16-vs-f32 noise; argmax matches HF). This proves the *actual ggml ops* — NEOX
`ggml_rope_ext` with proportional `freq_factors`, cross-attention via the build_attn_mha permute
layout (q [hd,n_head,q]; external K/V [hd,nkv,kvlen]; GQA broadcast; scale 1.0; no mask), gemma
sandwich norms, gelu, end-of-layer layer_scalar. replay.cpp is the exact op-sequence reference
for build_arch_graph. NOTE: build_arch_graph must obtain the external KV length + values + the
2*backbone embd from the speculative driver (a shared side-channel), so the graph and the driver
are co-designed; the standalone harness gates the math without that plumbing.

**build_arch_graph + input plumbing DONE & VERIFIED through the real llama API.**
`src/models/gemma4-assistant.cpp` now implements the forward (transcribed from replay.cpp);
`llama_gemma4_assistant_set_io()` (src/llama-ext.h) attaches the driver's host buffers (wide
embd + external full/swa KV + kv_len) to the model, and `llm_graph_input_gemma4_assistant`
copies them into the graph. `res->t_embd` is the post-norm hidden (n_embd); the driver applies
mtp.post_projection on the host. Gate `devtools/gemma4_assistant/test_decode.cpp` loads the
draft, attaches the oracle inputs, runs `llama_decode`, and matches HF to rel ~9e-4 on logits +
hidden (argmax 17887 == HF). Reserve-time (io==null) falls back to kv_len=n_ctx for sizing.
REMAINING: the speculative driver + target-KV extraction.

**Implementation steps:**
1. Inference graph (`src/models/gemma4-assistant.cpp::graph`): input embd width is 2*backbone;
   `pre_projection` → per layer {attn_norm → Q=q_proj, q_norm, RoPE (proportional for full /
   default for swa) → cross-attention over provided K/V (GQA: 32 Q heads over 16 swa / 4 full
   KV heads, head_dim 256/512) with a bidirectional mask → wo, layer_output_scale, post_attn
   norm, residual → ffn (gelu, pre/post norm), residual} → output_norm → both `post_projection`
   (as embeddings output) and `lm_head` logits. NOTE: backbone K/V must arrive as graph inputs.
2. KV/hidden/embedding extraction API (new `llama-ext.h` surface): read the target's K/V for the
   two layer types and its final hidden state; expose target `get_rows` on tok_embd (or run a
   tiny target embed). This is the bulk of the novel infrastructure.
3. `common/common.h`: add `COMMON_SPECULATIVE_TYPE_DRAFT_GEMMA4_ASSISTANT`.
4. `common/speculative.cpp`: impl modeled on `common_speculative_impl_draft_mtp` (`:409`),
   implementing the loop above (`process`/`draft`/`accept`, `need_embd_pre_norm()`), feeding the
   draft `ctx_dft` via `batch.embd` of width 2*backbone and injecting the fixed backbone KV.
5. Wire CLI/server flags so `draft-gemma4-assistant` is selectable with `--model` (Gemma 4
   backbone) + `--model-draft` (this GGUF).

### Phase D — Quantize to NVFP4
Deliverable: NVFP4 GGUF that loads and runs on Blackwell.

Largely free once Phase A/B land (arch-agnostic path). Tasks: confirm `pre_projection`,
`post_projection`, `centroids`, and backbone weights are NVFP4-eligible (row width % 64 == 0,
else Q8_0 fallback); ensure `token_ordering` (index buffer) and tied embeddings are excluded
from quantization; spot-check `llama-quantize <f16> <nvfp4> NVFP4` output loads.

## Key risks / open questions
1. **Shared target KV (highest risk).** The reference draft attends over the *backbone's* KV
   states, not its own. MTP/EAGLE feed hidden states but compute their own KV. Faithful option:
   new plumbing to inject external K/V tensors into the draft's `build_attn`. Fallback: have the
   draft recompute KV from fed hidden states (diverges from reference; likely lowers acceptance).
2. **Exact `inputs_embeds` composition. — RESOLVED.** It is
   `concat(target_embed(last_token_id), last_hidden_state)` (both backbone-dim, in that order),
   per `generation/candidate_generator.py:1379`. See the resolved data flow in Phase C.
3. **Bidirectional + flipped-SWA masks.** llama.cpp masks are causal; the draft needs new mask
   construction (`create_bidirectional_*` + the SWA kv-axis flip in `create_attention_masks`).
4. **Centroid head in ggml.** topk → gather → scatter-to-full-vocab with a per-position fill is
   expressible but must fit static-shape graph constraints; verify numerically vs reference.
5. **Acceptance/sampling loop** integration (backend sampling, rollback on partial accept) per
   the MTP impl's bookkeeping.

## File checklist
- `conversion/__init__.py` — register arch name → module.
- `conversion/gemma.py` — `Gemma4AssistantModel` converter.
- `gguf-py/gguf/constants.py` (+ writer helpers) — arch enum, tensors, KV keys.
- `src/llama-arch.h` / `src/llama-arch.cpp` — `LLM_ARCH_GEMMA4_ASSISTANT`, name, new `LLM_KV_*`
  and `LLM_TENSOR_*`, tensor-name map.
- `src/llama-hparams.h` — `backbone n_embd`, `n_centroids`, `centroid_top_k`, ordered-embd flag.
- `src/models/models.h` — `struct llama_model_gemma4_assistant`.
- `src/models/gemma4-assistant.cpp` (new) — hparams/tensors/graph.
- `src/llama-model.cpp` — factory case, `rope_type` (NEOX), all-shared-KV handling.
- `src/llama-graph.cpp` (+ `src/llama-ext.h`) — external-KV attention + bidirectional masks.
- `common/common.h`, `common/speculative.cpp` — new draft type + impl.
- CLI/server flag wiring for selecting the draft type.

## Verification
1. **Convert**: `python convert_hf_to_gguf.py <model_dir> --outtype f16`; inspect with
   `llama-gguf` / `gguf_dump.py` — confirm `general.architecture == gemma4-assistant`, all
   tensors (pre/post proj, centroids, token_ordering) and new KV keys present.
2. **Quantize**: `llama-quantize <f16>.gguf <nvfp4>.gguf NVFP4`; confirm it loads via
   `llama-cli`/loader without missing-tensor errors.
3. **Numerical parity**: for a fixed `(inputs_embeds, shared_kv_states)`, dump draft logits and
   compare against the HF `Gemma4AssistantForCausalLM` reference within tolerance.
4. **End-to-end speculative**: run a Gemma 4 backbone with the draft type selected; verify
   generated text matches the backbone-alone output (lossless speculative decoding) and report
   acceptance rate + tokens/s speedup on the Blackwell GPU.
5. Build with CUDA Blackwell enabled; confirm native FP4 kernels are used (no CPU dequant
   fallback) for the NVFP4 weights.
