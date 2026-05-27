#include "models.h"

// Gemma 4 Assistant: a speculative-decoding draft head for a Gemma 4 backbone.
//
// Architecture notes (see google/gemma-4-*-it-assistant):
//   - Dense Gemma-4 text stack: no MoE, no per-layer embeddings, no double-wide MLP.
//   - num_kv_shared_layers == num_hidden_layers: every layer reuses the backbone KV.
//   - The draft has NO k/v projections (only attn_q + attn_q_norm); attention runs over the
//     backbone's shared KV states (cross-attention), so it cannot run standalone. The inference
//     graph therefore lives behind the speculative path and is not yet implemented here.
//   - Input is the backbone hidden state projected through mtp.pre_projection
//     (2*n_embd_backbone -> n_embd); output is mtp.post_projection (n_embd -> n_embd_backbone)
//     plus next-token logits via the tied lm_head (or a centroid head when use_ordered_embeddings).

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    // all layers share the backbone KV
    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);
    hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t) n_kv_shared_layers;

    hparams.f_attention_scale = 1.0f; // matches Gemma4 (self.scaling = 1.0)

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);

    // assistant-specific metadata
    ml.get_key(LLM_KV_N_EMBD_BACKBONE,            hparams.n_embd_backbone);
    ml.get_key(LLM_KV_N_CENTROIDS,                hparams.n_centroids,            false);
    ml.get_key(LLM_KV_CENTROID_TOP_K,             hparams.centroid_top_k,         false);
    ml.get_key(LLM_KV_USE_ORDERED_EMBEDDINGS,     hparams.use_ordered_embeddings, false);
    ml.get_key(LLM_KV_ATTENTION_K_EQ_V,           hparams.attn_k_eq_v,            false);

    type = LLM_TYPE_UNKNOWN; // small draft head (e.g. ~470M for the 31B target)
}

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        // tied word embeddings
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // backbone <-> draft projections (model level, no block id)
    const int64_t n_embd_bb = hparams.n_embd_backbone;
    mtp_pre_proj  = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJ,  "weight"), {2*n_embd_bb, n_embd},  0);
    mtp_post_proj = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJ, "weight"), {n_embd,      n_embd_bb}, 0);

    if (hparams.use_ordered_embeddings) {
        mtp_centroids      = create_tensor(tn(LLM_TENSOR_MTP_CENTROIDS,      "weight"), {n_embd, (int64_t) hparams.n_centroids}, 0);
        mtp_token_ordering = create_tensor(tn(LLM_TENSOR_MTP_TOKEN_ORDERING, "weight"), {n_vocab}, 0);
    }

    // single, global rope_freqs used by the full-attention (proportional rope) layers.
    // the "rope_freqs" name has no block placeholder, so it always resolves to the one
    // global tensor; the bid only satisfies the repeating-tensor check, and TENSOR_DUPLICATED
    // avoids reloading it for subsequent full layers.
    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_i      = hparams.n_head(i);
        const int64_t n_embd_head_i = hparams.n_embd_head_k(i);
        const int64_t n_ff_i        = hparams.n_ff(i);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);

        // no k/v projections: the draft attends over the backbone's KV
        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), {n_embd, n_embd_head_i * n_head_i}, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_embd_head_i * n_head_i, n_embd}, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head_i}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, 0);

        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.ffn_gate       = create_tensor(tn(LLM_TENSOR_FFN_GATE,       "weight", i), {n_embd,  n_ff_i}, 0);
        layer.ffn_up         = create_tensor(tn(LLM_TENSOR_FFN_UP,         "weight", i), {n_embd,  n_ff_i}, 0);
        layer.ffn_down       = create_tensor(tn(LLM_TENSOR_FFN_DOWN,       "weight", i), {n_ff_i,  n_embd}, 0);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, 0);

        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head_i / 2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }
    }
}

llama_model_gemma4_assistant::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params), model(model) {
    // The draft head cross-attends over the backbone's shared KV states, which are not yet
    // plumbed through llama.cpp's graph inputs. Inference is wired up in the speculative path
    // (COMMON_SPECULATIVE_TYPE_DRAFT_GEMMA4_ASSISTANT). Until then, decoding is unsupported.
    throw std::runtime_error(
        "gemma4_assistant: inference graph not implemented yet "
        "(draft head requires cross-attention over the backbone KV via the speculative path)");
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
