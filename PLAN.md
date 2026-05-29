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
1. Inference graph (`build_arch_graph`): ✅ DONE & GPU-verified (see above).
2. Input plumbing (`llama_gemma4_assistant_set_io` + graph input): ✅ DONE & verified.

### Phase C driver — DE-RISKED DESIGN (remaining work; GPU-iterated)
Every mechanism is identified and the verifiable ones are checked. Implement in
`common/speculative.cpp` as a new impl + type (`COMMON_SPECULATIVE_TYPE_DRAFT_GEMMA4_ASSISTANT`
in common.h; update the name map, `type_to_str`, the `static_assert(...==9)`→10, the enable
logic ~`:1330`, and the dispatch switch ~`:1386`). The AR loop (per the resolved data flow):

- **Seed** (in `process()`, after the target decodes): `last_hidden` = target final hidden of
  the last validated token via `llama_get_embeddings_pre_norm_ith` (reuse the MTP path; set
  `need_embd_pre_norm()`=true). VALIDATE on GPU that this equals HF `hidden_states[-1]` (pre- vs
  post-final-norm); if it's the post-norm one, use `llama_get_embeddings_ith`.
- **Target shared KV** — CONFIRMED against the live 31B via `devtools/gemma4_assistant/probe_kv.cpp`.
  A `cb_eval` callback (llama_context_params.cb_eval) on the target captures the backbone's
  `Kcur_pos` (post-RoPE K) and `Vcur_normed` (normed V). For google-gemma-4-31B (60 layers,
  shared_kv_layers=0, full-attn every 6th): the shared KV = the **last full layer (59)** →
  `Kcur_pos-59`/`Vcur_normed-59` `[512,4,seq]` (io.k_full/v_full) and the **last sliding layer
  (58)** → `Kcur_pos-58`/`Vcur_normed-58` `[256,16,seq]` (io.k_swa/v_swa). Shapes match the
  draft's external KV exactly; used as-is (no extra rope/norm). Generalize the layer indices
  from `sliding_window_pattern` (last False = full, last True = sliding). cb_eval is set at
  target-context creation, so the server must create ctx_tgt with this callback (setup wrinkle).
  **IMPORTANT nuance the probe revealed**: `Kcur_pos` holds only the *current batch's* tokens;
  the draft needs the *full-sequence* KV (in the target's KV cache). So either (a) read the KV
  cache for layers 58/59, or (b) do a full-sequence target forward to capture them (matches how
  HF assembles shared_kv_states; simpler but re-prefills each cycle).
- **Target embed**: each draft step needs `target_embed(last_token)` (backbone 5376-dim). Add a
  small API to read backbone `token_embd` rows (get_rows) into host, or fold via a tiny target
  graph. Concat with `last_hidden` → io.embd (width 2*backbone).
- **Draft step**: `llama_gemma4_assistant_set_io(draft, &io)` → `llama_decode(ctx_dft, batch)`
  (batch carries position=[len-1+i]; token unused) → `logits` (argmax = next draft token) →
  embeddings = post-norm hidden (1024) → host `post_projection` (VERIFIED: `nrm @ post.T`,
  rel 4e-7; for an NVFP4 draft, dequantize post_projection first, or run the draft at f16/Q8_0
  since it's only ~940MB) → next `last_hidden` (5376). Repeat K times.
3. Wire CLI/server flags so `draft-gemma4-assistant` is selectable with `--model` (Gemma 4
   backbone) + `--model-draft` (this GGUF), incl. creating ctx_tgt with the KV-capture cb_eval.

Verified driver building blocks: draft decode (GPU, rel ~1.7e-3), host post_projection
(rel 4e-7). Remaining is the framework integration + target KV/embed extraction, validated
end-to-end by acceptance rate / lossless output against the live 31B target.

### Phase C driver — END-TO-END LOSSLESS RUN ACHIEVED (option-b)
`devtools/gemma4_assistant/spec_run.cpp` runs greedy speculative decoding (target 31B + draft)
and is verified **lossless** (output == target-only greedy) on the Blackwell GPU. Confirmed the
full pipeline: cb_eval KV capture (layers 59/58) + fixed-position AR draft loop + host
post_projection feedback + target verify/accept. Key finding: the seed/feedback hidden must be
**POST-norm** (HF hidden_states[-1]); fixing this raised acceptance 5%→12.5%.
### Phase C driver — OPTION (a) DONE (incremental, lossless) + acceptance findings
spec_run.cpp now uses the efficient path: incremental target decode + host-side accumulation of
the shared KV (append each cycle's captured Kcur_pos/Vcur_normed; roll back rejected drafts via
llama_memory_seq_rm). Still **lossless**. Default K=2 (k>=2 almost never accepts; K=2 drafts 38
vs 76 at K=4 for the same output).

Per-k breakdown (NVFP4 target + f16 draft): k=0 ~32%, k>=1 ~5%; ~1.3 tok/cycle.
Key findings:
- Acceptance is **fundamentally limited by the NVFP4 target**: the draft was trained on the
  full-precision backbone and is sensitive to small perturbations in the hidden/KV it consumes.
- Incremental KV != full-forward KV (max ~0.08 on K), independent of KV-cache dtype — it's the
  NVFP4 matmul/attention using different kernels for single-token vs batched decode. So option-b's
  earlier 53% k=0 was a **full-forward artifact**; option-a's ~32% is the realistic deployment
  number (the target always decodes incrementally in production).
- A higher-precision target would lift acceptance but Q8_0/f16 31B doesn't fit in 32GB.

KNOWN LIMITATION: the SWA layer's accumulated KV isn't windowed; correct only while
seq_len <= sliding_window (1024). For longer contexts, feed only the last 1024 SWA positions
(needs separate kv_len for full vs swa in the io/graph).

### Phase C driver — llama-server integration WORKING
`COMMON_SPECULATIVE_TYPE_DRAFT_GEMMA4_ASSISTANT` ("draft-gemma4-assistant") in
common/speculative.cpp runs end-to-end under llama-server (single-seq), lossless, at ~20-33%
acceptance (matching the NVFP4-limited standalone). Run:
`--model <31B> --model-draft <assistant f16> --spec-type draft-gemma4-assistant --parallel 1`.
Enablers added: `llama_set_eval_callback` (persists cb_eval in cparams; the decode path re-applies
it each build) to capture the backbone shared KV on the server's ctx_tgt; target GGUF path
threaded through `common_params_speculative_draft`; draft-simple auto-enable suppressed when this
type is selected; ctx_dft embeddings enabled for the host post_projection.

Long-context memory (DONE): sliding-attention layers are fed only the last `sliding_window`
(1024) KV positions (io split into kv_len_full/kv_len_swa; no SWA mask needed), and KV inputs to
the draft graph are F16. This makes 128K both correct and fit (SWA KV input collapses from ~GiB
to ~MiB; full-layer KV input ~1 GiB f16). sliding_window hardcoded 1024 (TODO: read hparams.n_swa).

### Next steps for 128K / 256K (priority order)
Verified: 128K loads + runs + fits at ~25 GiB / 32 GiB. The remaining work is about making long
context fast (and 256K fit), then acceptance + productionization.

1. **On-device resident KV (critical path for 128K/256K perf). ✅ DONE (89fefbcea); needs GPU
   validation.** The driver was memcpy'ing the full-layer KV host->device on EVERY draft step
   (~0.5-1 GiB at 128K, PCIe-bound). Now a persistent device buffer (`dev_k_full`/`dev_v_full`,
   `[hd,nkv,n_ctx]` f16, lazily allocated on the draft device via `llama_context_dev_buft`) holds
   the full-layer KV; the draft graph *views* it (`ggml_view_3d` over the first kv_len_full
   positions — no per-decode copy). `commit(npos)` appends only the new validated positions
   (f32->f16) each cycle; SWA KV stays a small windowed host feed (kv_len_swa <= 1024). io carries
   the device tensor handles (`dev_k_full`/`dev_v_full`); the host `k_full`/`v_full` remain a
   fallback (spec_run still uses them, verified non-regressed/lossless). REMAINING: validate on
   GPU/server at 128K — confirm load (look for the device-KV alloc), lossless output, and that
   long-context tok/s improves vs the host-copy version; watch the view-of-external-tensor + sched
   interaction (the one untested path).
2. **256K bring-up.** Target KV ~doubles (~9 GiB q8/q5) -> target+KV ~26 GiB, so 256K likely needs
   `--cache-type q4_0` (~6 GiB). Confirm load at `--ctx-size 262144` and tune KV-quant/quality.
   Note dev_k_full now also scales with n_ctx (~1 GiB f16 at 256K) — budget it.
3. **Capture-gap fix (long-context correctness). ✅ DONE (f2af48229).** The decode graph-reuse path
   now re-applies `cparams.cb_eval` via `ggml_backend_sched_set_eval_callback`, so the eval
   callback fires on every ubatch (incl. reused/reserved graphs) and the full-layer KV no longer
   has a gap at pos0/long prompts.
4. **Acceptance: id_last/last_hidden off-by-one** (biggest acceptance lever; see below).
5. **Multi-sequence / continuous batching** (currently single-seq; per-seq state keyed by seq_id).
6. **Cleanups**: read sliding_window from hparams (not hardcoded 1024); extend the HF oracle to a
   >1024-token case to numerically validate SWA windowing before trusting 256K.

Remaining acceptance levers (all near the NVFP4 ceiling; refinement, not blockers):
- ~~first prompt ubatch (pos0==0) captures no KV on reused/reserved graphs~~ ✅ FIXED [step 3].
- id_last/last_hidden off-by-one: the server samples the bonus then drafts immediately, so the
  draft pairs embed(id_last) with the previous token's hidden (EAGLE/MTP-style) vs the assistant's
  same-position training. Fully matching it needs forwarding id_last through the target first. [step 4]
- multi-sequence/continuous-batching (currently single-seq). [step 5]

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

### Phase E — KV-cache backfill (caching + speculation together)  [SCOPED, not implemented]

**Context.** The driver builds the backbone shared KV only from the `cb_eval` callback, which
fires for *decoded* tokens. When the server serves a prompt prefix from its prompt-cache
(`--cache-ram`) or context-checkpoint (`--ctx-checkpoints`), those positions are restored into
the target KV **without** being decoded, so `cb_eval` misses them and `acc_len` lags the target
(`pos0 > acc_len`). Today the driver detects this and *pauses* drafting (commit `6db18160a`) —
correct + lossless but no speedup until a fresh prompt. Workaround: run with `--cache-ram 0
--ctx-checkpoints 0`. This phase makes caching + speculation coexist by backfilling the missing
KV from the target's own cache.

**Verified premise.** `build_attn` stores into the cache exactly the tensors `cb_eval` captures:
`Kcur_pos` (post-RoPE, post-k-norm K) and `Vcur_normed` (rms-normed V), `src/models/gemma4.cpp:226-239`.
So the cache *is* the right source; we just need to read it.

**Approach.** On a gap (`pos0 > acc_len`) in `process()`, read the target KV cache for the shared
layers over `[acc_len, pos0)`, dequantize to f32, and `commit()` it (advancing `acc_len` to `pos0`);
the current decode's `cb_eval` capture then commits on top. Drafting resumes aligned.

**Components.**
1. **Staging read API** (`src/llama-ext.h` + `src/llama-context.cpp`):
   `int llama_kv_read_layer_f32(ctx, int il, llama_seq_id, llama_pos p0, llama_pos p1, float * k_out, float * v_out)`.
   Implemented against the memory/`llama_kv_cache` internals (model the copy on `state_write_data`,
   `src/llama-kv-cache.cpp:1969-2066`). Returns #positions read (may be < p1-p0 for SWA layers
   whose window has scrolled past). Per position: K row = `[n_embd_head_k, n_head_kv]`, V row =
   `[n_embd_head_v, n_head_kv]` in ne-order — already the layout the draft `io`/`commit()` expects.
2. **Driver backfill** (`common/speculative.cpp`): in `process()` replace the `pos0 > acc_len` warn
   with a backfill: call the API for `layer_full` and `layer_swa` over `[acc_len, pos0)`, f32→f16,
   `commit(npos, k_full, v_full, k_swa, v_swa)`. SWA only needs the last `sliding_window`; cap reads.

**Obstacles (ranked).**
- **iSWA routing.** Target uses an iSWA cache: `layer_full` (59) lives in the base cache, `layer_swa`
  (58) in the SWA sub-cache (`src/llama-kv-cache-iswa.cpp`). The API must pick the sub-cache via
  `hparams.is_swa(il)`. The SWA sub-cache only holds ~`n_swa` recent positions — fine (that's all we
  feed) but reads of older gap positions for the SWA layer will (correctly) return fewer rows.
- **Layer remap.** Use `map_layer_ids[il]` (`src/llama-kv-cache.cpp:224,247`) to get the physical
  cache-layer index (shared-KV layers reuse earlier slots).
- **Dequant.** Cache dtype = `cache_type_k/v` (q8_0/q5_1/…); copy raw via `ggml_backend_tensor_get`
  then dequantize with the ggml row dequantizers. (cb_eval was full-precision f32 pre-quant, so
  backfilled positions are ~1 quant-step noisier — acceptable for the restored prefix.)
- **V transpose.** If `v_trans` (no-FA path), V is `[kv_size, …]` per element → gather per embd dim
  (`src/llama-kv-cache.cpp:1189-1194,2059`); if FA/`!v_trans`, V row is contiguous like K. Handle both.
- **pos→cell lookup.** No direct finder; one O(kv_size) pass over `v_cells[stream]`
  (`src/llama-kv-cells.h`) to build a `pos→cell` map for the gap range (one-time per gap, cheap).

**Effort/risk.** ~1–2 days. Main risk is the read API correctness (layout/transpose/dequant/iSWA);
de-risk by a unit gate that captures one position via *both* `cb_eval` and the read API and asserts
they match to ~1e-2 (f32 vs quant). Then end-to-end: acceptance with caching ON should match the
`--cache-ram 0 --ctx-checkpoints 0` baseline (~14%).

**Alternative (cheaper, considered & rejected):** force re-decode of the gap — defeats the caching
speedup and the driver can't drive the server's decode loop. The cache read is the only way to keep
both.
