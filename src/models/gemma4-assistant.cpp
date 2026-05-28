#include "models.h"
#include "../llama-ext.h"

#include <algorithm>
#include <cstring>

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
    // optional NVFP4 weight_scale_2 companions (the generic scale pass only covers per-layer tensors)
    mtp_pre_proj_s  = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJ,  "scale"), {1}, TENSOR_NOT_REQUIRED);
    mtp_post_proj_s = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJ, "scale"), {1}, TENSOR_NOT_REQUIRED);

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

void llm_graph_input_gemma4_assistant::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);
    if (io == nullptr) {
        return;
    }
    // input graph tensors live in a host-mapped buffer; copy the driver's host data into them
    auto cp = [](ggml_tensor * t, const float * src) {
        if (t && src) {
            GGML_ASSERT(ggml_backend_buffer_is_host(t->buffer));
            memcpy(t->data, src, ggml_nbytes(t));
        }
    };
    cp(embd,   io->embd);
    cp(k_full, io->k_full);
    cp(v_full, io->v_full);
    cp(k_swa,  io->k_swa);
    cp(v_swa,  io->v_swa);
}

// Forward graph, transcribed from the numpy- and ggml-verified reference
// (devtools/gemma4_assistant/{numpy_reference.py,replay.cpp}, which match HF to rel ~1e-3).
llama_model_gemma4_assistant::graph::graph(const llama_model & model_, const llm_graph_params & params)
    : llm_graph_context(params), model(model_) {
    const auto & m = static_cast<const llama_model_gemma4_assistant &>(model_);

    const llama_gemma4_assistant_io * io = m.io;

    const int64_t n_embd_bb = hparams.n_embd_backbone;
    // io is null during the context's graph_reserve (buffer sizing); fall back to n_ctx so the
    // reserved graph covers the worst case. set_input no-ops when io is null. Real decodes attach
    // io (via llama_gemma4_assistant_set_io) and the graph is rebuilt with the actual kv_len.
    const int64_t kv_len = (io && io->kv_len > 0) ? io->kv_len : std::max<int64_t>(cparams.n_ctx, 1);

    // representative swa / full layers (for the external KV head dims)
    int idx_swa = -1, idx_full = -1;
    for (int il = 0; il < n_layer; ++il) {
        if (hparams.is_swa(il)) { if (idx_swa  < 0) idx_swa  = il; }
        else                    { if (idx_full < 0) idx_full = il; }
    }

    // inputs provided by the speculative driver (see llama_gemma4_assistant_set_io)
    auto inp = std::make_unique<llm_graph_input_gemma4_assistant>(io);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2*n_embd_bb, n_tokens);
    ggml_set_input(inp->embd);
    if (idx_full >= 0) {
        inp->k_full = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams.n_embd_head_k_full, hparams.n_head_kv(idx_full), kv_len);
        inp->v_full = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams.n_embd_head_k_full, hparams.n_head_kv(idx_full), kv_len);
        ggml_set_input(inp->k_full);
        ggml_set_input(inp->v_full);
    }
    if (idx_swa >= 0) {
        inp->k_swa = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams.n_embd_head_k_swa, hparams.n_head_kv(idx_swa), kv_len);
        inp->v_swa = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams.n_embd_head_k_swa, hparams.n_head_kv(idx_swa), kv_len);
        ggml_set_input(inp->k_swa);
        ggml_set_input(inp->v_swa);
    }
    auto * INP = (llm_graph_input_gemma4_assistant *) res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();

    // pre_projection: [2*n_embd_backbone] -> [n_embd]
    ggml_tensor * x = build_lora_mm(m.mtp_pre_proj, INP->embd, m.mtp_pre_proj_s);
    cb(x, "pre_projection", -1);

    for (int il = 0; il < n_layer; ++il) {
        const bool    swa   = hparams.is_swa(il);
        const int64_t hd    = hparams.n_embd_head_k(il);
        const int64_t nh    = hparams.n_head(il);
        const int     n_rot = hparams.n_rot(il);
        const float   fb    = m.get_rope_freq_base (cparams, il);
        const float   fs    = m.get_rope_freq_scale(cparams, il);
        ggml_tensor * ff    = swa ? nullptr     : m.layers[il].rope_freqs; // proportional rope (full layers)
        ggml_tensor * K     = swa ? INP->k_swa  : INP->k_full;
        ggml_tensor * V     = swa ? INP->v_swa  : INP->v_full;

        ggml_tensor * res_x = x;

        ggml_tensor * h = build_norm(x, m.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);

        // Q only (no K/V projections); cross-attend over the backbone's KV
        ggml_tensor * Q = build_lora_mm(m.layers[il].wq, h, m.layers[il].wq_s);
        Q = ggml_reshape_3d(ctx0, Q, hd, nh, n_tokens);
        Q = build_norm(Q, m.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Q = ggml_rope_ext(ctx0, Q, inp_pos, ff, n_rot, rope_type, n_ctx_orig, fb, fs,
                          ext_factor, attn_factor, beta_fast, beta_slow);

        ggml_tensor * q_p = ggml_permute(ctx0, Q, 0, 2, 1, 3);                  // [hd, q, nh]
        ggml_tensor * k_p = ggml_permute(ctx0, K, 0, 2, 1, 3);                  // [hd, kv_len, nkv]
        ggml_tensor * kq  = ggml_mul_mat(ctx0, k_p, q_p);                       // [kv_len, q, nh] (GQA bcast)
        kq = ggml_soft_max(ctx0, kq);                                          // scale 1.0, no mask (q_len==1)
        ggml_tensor * v_p = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // [kv_len, hd, nkv]
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v_p, kq);                       // [hd, q, nh]
        kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);                             // [hd, nh, q]
        ggml_tensor * attn = ggml_cont_2d(ctx0, kqv, hd*nh, n_tokens);
        ggml_tensor * o = build_lora_mm(m.layers[il].wo, attn, m.layers[il].wo_s);

        o = build_norm(o, m.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * sa = ggml_add(ctx0, res_x, o);

        ggml_tensor * f = build_norm(sa, m.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        f = build_ffn(f,
                m.layers[il].ffn_up,   nullptr, m.layers[il].ffn_up_s,
                m.layers[il].ffn_gate, nullptr, m.layers[il].ffn_gate_s,
                m.layers[il].ffn_down, nullptr, m.layers[il].ffn_down_s,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        f = build_norm(f, m.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, il);

        x = ggml_add(ctx0, sa, f);
        if (m.layers[il].out_scale) {
            x = ggml_mul(ctx0, x, m.layers[il].out_scale); // layer_scalar at end of layer
        }
        cb(x, "l_out", il);
    }

    ggml_tensor * nrm = build_norm(x, m.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(nrm, "result_norm", -1);
    // expose the post-norm hidden state as embeddings; the driver applies mtp.post_projection
    // on the host to obtain the next backbone-space hidden state.
    res->t_embd = nrm;

    ggml_tensor * logits = build_lora_mm(m.output, nrm, m.output_s);
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, nrm);
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

void llama_gemma4_assistant_set_io(llama_model * model, const llama_gemma4_assistant_io * io) {
    if (model && model->arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        static_cast<llama_model_gemma4_assistant *>(model)->io = io;
    }
}
