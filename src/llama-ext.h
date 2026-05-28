#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

//
// pre-norm embeddings (hidden state before the final output norm)
//

// Set whether the context outputs pre-norm embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value, bool masked);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_pre_norm    (struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_pre_norm_ith(struct llama_context * ctx, int32_t i);

//
// gemma4_assistant speculative draft I/O (staging)
//
// The Gemma 4 Assistant draft head cross-attends over the *backbone's* KV states and consumes
// projected backbone hidden states as input. The speculative driver fills these host buffers
// from the target model and attaches them to the draft model before each llama_decode of the
// draft. All pointers must stay valid until the decode completes. Layouts are ggml ne-order
// (ne0 fastest). kv_len is the backbone sequence length being attended over.
//
// The graph outputs the post-norm hidden state (n_embd-wide) via the embeddings output; the
// driver applies mtp.post_projection on the host to obtain the next backbone-space hidden state.
struct llama_gemma4_assistant_io {
    int32_t kv_len   = 0;
    int32_t n_tokens = 0;
    const float * embd   = nullptr; // [2*n_embd_backbone, n_tokens]
    const float * k_full = nullptr; // [n_embd_head_k,     n_head_kv_full, kv_len]
    const float * v_full = nullptr; // [n_embd_head_k,     n_head_kv_full, kv_len]
    const float * k_swa  = nullptr; // [n_embd_head_k_swa, n_head_kv_swa,  kv_len]
    const float * v_swa  = nullptr; // [n_embd_head_k_swa, n_head_kv_swa,  kv_len]
};

// attach (or clear, with io == nullptr) draft I/O to a gemma4_assistant model.
// no-op for models of any other architecture.
LLAMA_API void llama_gemma4_assistant_set_io(struct llama_model * model, const struct llama_gemma4_assistant_io * io);
