// End-to-end speculative-decoding loop for the Gemma 4 Assistant draft (option-b KV capture).
//
// Verifies the full pipeline: target (Gemma-4-31B) + draft (assistant) running greedy
// speculative decoding, and checks LOSSLESSNESS (speculative output == target-only greedy).
//
// Option-b KV capture: each target forward re-processes the full sequence (KV cache cleared),
// so a cb_eval callback capturing the backbone's last full/sliding layer Kcur_pos/Vcur_normed
// yields the full-sequence shared KV the draft cross-attends over. Slow (re-prefill per cycle)
// but correct; option-a (KV-cache read) is the later perf swap.
//
// build:
//   g++ -O2 -std=c++17 devtools/gemma4_assistant/spec_run.cpp -o /tmp/g4a_spec \
//       -Iinclude -Isrc -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$PWD/build/bin
// run:
//   /tmp/g4a_spec <target.gguf> <draft_f16.gguf> [n_predict] [k_draft]

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// --- backbone (31B) layer indices carrying the shared KV (from sliding_window_pattern:
//     last full-attention layer / last sliding-attention layer). For google-gemma-4-31B. ---
static int LAYER_FULL = 59;
static int LAYER_SWA  = 58;

static const int N_EMBD_BB = 5376; // backbone hidden
static const int HD_FULL = 512, NKV_FULL = 4;
static const int HD_SWA  = 256, NKV_SWA  = 16;
static const int N_EMBD_DRAFT = 1024;
static const int SLIDING_WINDOW = 1024; // sliding-attention layers see only the last this-many KV positions

// ---------- captured backbone KV (host) ----------
struct KVCap {
    int seq = 0;
    std::vector<float> k_full, v_full, k_swa, v_swa; // [hd, nkv, seq] ggml-order
    void want(struct ggml_tensor * t) {
        const char * n = t->name;
        auto grab = [&](const char * name, std::vector<float> & dst) {
            if (strcmp(n, name) == 0) {
                dst.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
                seq = (int) t->ne[2];
            }
        };
        char b[64];
        snprintf(b, sizeof(b), "Kcur_pos-%d",     LAYER_FULL); grab(b, k_full);
        snprintf(b, sizeof(b), "Vcur_normed-%d",  LAYER_FULL); grab(b, v_full);
        snprintf(b, sizeof(b), "Kcur_pos-%d",     LAYER_SWA);  grab(b, k_swa);
        snprintf(b, sizeof(b), "Vcur_normed-%d",  LAYER_SWA);  grab(b, v_swa);
    }
};

static KVCap * g_cap = nullptr;
static bool cb_eval(struct ggml_tensor * t, bool ask, void *) {
    const char * n = t->name;
    const bool want = strncmp(n, "Kcur_pos-", 9) == 0 || strncmp(n, "Vcur_normed-", 12) == 0;
    if (ask) return want;
    if (want && g_cap) g_cap->want(t);
    return true;
}

// ---------- read a named tensor's raw bytes from a GGUF (no full-file load) ----------
struct RawTensor {
    std::vector<uint8_t> bytes;
    enum ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1,1,1,1};
};

static RawTensor load_raw(const std::string & path, const std::string & name) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params p = { /*no_alloc=*/ true, /*ctx=*/ &ctx }; // metadata only
    struct gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) { fprintf(stderr, "gguf open failed: %s\n", path.c_str()); exit(1); }
    int64_t idx = gguf_find_tensor(g, name.c_str());
    if (idx < 0) { fprintf(stderr, "tensor %s not in %s\n", name.c_str(), path.c_str()); exit(1); }
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    RawTensor r; r.type = t->type;
    for (int i = 0; i < 4; ++i) r.ne[i] = t->ne[i];
    const size_t nb  = ggml_nbytes(t);
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, idx);
    r.bytes.resize(nb);
    FILE * f = fopen(path.c_str(), "rb");
    fseek(f, (long) off, SEEK_SET);
    if (fread(r.bytes.data(), 1, nb, f) != nb) { fprintf(stderr, "short read %s\n", name.c_str()); exit(1); }
    fclose(f);
    gguf_free(g);
    ggml_free(ctx);
    return r;
}

// dequantize row `row` (length ne0) of a 2D tensor to f32 (supports F32/F16/Q8_0)
static void get_row_f32(const RawTensor & r, int64_t row, std::vector<float> & out) {
    const int64_t ne0 = r.ne[0];
    out.resize(ne0);
    if (r.type == GGML_TYPE_F32) {
        memcpy(out.data(), r.bytes.data() + (size_t) row * ne0 * 4, ne0 * 4);
    } else if (r.type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *) (r.bytes.data() + (size_t) row * ne0 * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < ne0; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
    } else if (r.type == GGML_TYPE_Q8_0) {
        // block_q8_0: { fp16 d; int8 qs[32]; } = 34 bytes, 32 weights/block
        const int    QK = 32, BS = 34;
        const int64_t nblk = ne0 / QK;
        const uint8_t * base = r.bytes.data() + (size_t) row * nblk * BS;
        for (int64_t b = 0; b < nblk; ++b) {
            const uint8_t * blk = base + b * BS;
            ggml_fp16_t d16; memcpy(&d16, blk, 2);
            const float d = ggml_fp16_to_fp32(d16);
            const int8_t * qs = (const int8_t *) (blk + 2);
            for (int i = 0; i < QK; ++i) out[b * QK + i] = d * qs[i];
        }
    } else {
        fprintf(stderr, "get_row_f32: unsupported type %s\n", ggml_type_name(r.type)); exit(1);
    }
}

static int argmax(const float * v, int n) { int a = 0; for (int i = 1; i < n; ++i) if (v[i] > v[a]) a = i; return a; }

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <target.gguf> <draft_f16.gguf> [n_predict] [k_draft]\n", argv[0]); return 1; }
    const std::string tpath = argv[1], dpath = argv[2];
    const int n_predict = argc > 3 ? atoi(argv[3]) : 24;
    const int K = argc > 4 ? atoi(argv[4]) : 2;

    llama_backend_init();

    // host copies of the cross-model weights the driver needs
    fprintf(stderr, "reading backbone token_embd + draft post_projection from GGUF...\n");
    RawTensor tok_embd = load_raw(tpath, "token_embd.weight");              // ne=[5376, vocab]
    RawTensor post_raw = load_raw(dpath, "mtp.post_projection.weight");     // ne=[1024, 5376]
    const int n_embd_bb = (int) tok_embd.ne[0];
    const int n_vocab   = (int) tok_embd.ne[1];
    // dequantize post_projection fully (small): post_f32[o*1024 + i] = W[o, i]
    std::vector<float> post_f32((size_t) post_raw.ne[1] * post_raw.ne[0]);
    { std::vector<float> rowbuf; for (int64_t o = 0; o < post_raw.ne[1]; ++o) { get_row_f32(post_raw, o, rowbuf); memcpy(post_f32.data() + (size_t) o * N_EMBD_DRAFT, rowbuf.data(), N_EMBD_DRAFT * sizeof(float)); } }

    // --- target ---
    llama_model_params tmp = llama_model_default_params(); tmp.n_gpu_layers = 99;
    llama_model * tgt = llama_model_load_from_file(tpath.c_str(), tmp);
    llama_context_params tcp = llama_context_default_params();
    tcp.n_ctx = 4096; tcp.n_batch = 1024; tcp.n_ubatch = 1024;
    tcp.cb_eval = cb_eval;
    tcp.embeddings = true; tcp.pooling_type = LLAMA_POOLING_TYPE_NONE; // post-norm hidden (HF hidden_states[-1])
    llama_context * ctx_tgt = llama_init_from_model(tgt, tcp);

    // --- draft ---
    llama_model_params dmp = llama_model_default_params(); dmp.n_gpu_layers = 99;
    llama_model * dft = llama_model_load_from_file(dpath.c_str(), dmp);
    llama_context_params dcp = llama_context_default_params();
    dcp.n_ctx = 256; dcp.n_batch = 8; dcp.n_ubatch = 8; dcp.embeddings = true; dcp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx_dft = llama_init_from_model(dft, dcp);

    const llama_vocab * vocab = llama_model_get_vocab(tgt);
    const char * prompt = "Q: What is the capital of France?\nA:";
    std::vector<llama_token> toks(64);
    int np = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks.data(), 64, true, false);
    toks.resize(np);

    // option-a: incremental target decode + host-side accumulation of the shared KV.
    // KV accumulators hold ggml layout [hd, nkv, pos] with pos slowest, so appending a decode's
    // captured Kcur_pos/Vcur_normed appends new positions to the full-sequence shared KV.
    KVCap cap;
    std::vector<float> acc_kf, acc_vf, acc_ks, acc_vs; int acc_len = 0;
    auto append_cap = [&]() {
        acc_kf.insert(acc_kf.end(), cap.k_full.begin(), cap.k_full.end());
        acc_vf.insert(acc_vf.end(), cap.v_full.begin(), cap.v_full.end());
        acc_ks.insert(acc_ks.end(), cap.k_swa.begin(),  cap.k_swa.end());
        acc_vs.insert(acc_vs.end(), cap.v_swa.begin(),  cap.v_swa.end());
        acc_len += cap.seq;
    };
    // decode `tk` at positions [base, base+n); the target cache must already hold [0, base).
    // captures the new positions' shared KV into `cap`; returns per-position logits + last hidden.
    auto decode_incr = [&](const std::vector<llama_token> & tk, int base,
                           std::vector<float> & out_logits, std::vector<float> & out_hidden) {
        g_cap = &cap; cap = KVCap{};
        const int n = (int) tk.size();
        llama_batch b = llama_batch_init(n, 0, 1);
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) { b.token[i] = tk[i]; b.pos[i] = base + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        if (llama_decode(ctx_tgt, b) != 0) { fprintf(stderr, "target decode failed\n"); exit(1); }
        out_logits.resize((size_t) n * n_vocab);
        for (int i = 0; i < n; ++i) { const float * lg = llama_get_logits_ith(ctx_tgt, i); memcpy(out_logits.data() + (size_t) i * n_vocab, lg, (size_t) n_vocab * sizeof(float)); }
        const float * h = llama_get_embeddings_ith(ctx_tgt, n - 1);
        out_hidden.assign(h, h + n_embd_bb);
        llama_batch_free(b);
    };

    // host helpers
    auto embed_bb = [&](llama_token t) {                       // raw backbone embedding row [5376]
        std::vector<float> out; get_row_f32(tok_embd, t, out); return out;
    };
    auto post_proj = [&](const float * nrm) {                  // [1024] -> [5376] : out = nrm @ W.T
        std::vector<float> out(n_embd_bb, 0.0f);
        for (int o = 0; o < n_embd_bb; ++o) {
            const float * w = post_f32.data() + (size_t) o * N_EMBD_DRAFT; // W[o, :]
            float s = 0.0f; for (int i = 0; i < N_EMBD_DRAFT; ++i) s += nrm[i] * w[i];
            out[o] = s;
        }
        return out;
    };

    // one draft step: returns next token, updates cur_h (post_projection output)
    llama_gemma4_assistant_io io{};
    auto draft_step = [&](llama_token cur_tok, std::vector<float> & cur_h, int kv_len, int pos) -> llama_token {
        std::vector<float> emb = embed_bb(cur_tok);            // [5376]
        std::vector<float> wide(2 * n_embd_bb);
        memcpy(wide.data(),               emb.data(),    (size_t) n_embd_bb * sizeof(float));
        memcpy(wide.data() + n_embd_bb,   cur_h.data(),  (size_t) n_embd_bb * sizeof(float));
        // build F16 feed buffers: full layers see all kv_len positions, sliding layers the last window
        const int kv_swa = std::min(kv_len, SLIDING_WINDOW);
        const int fpp_f = HD_FULL * NKV_FULL, fpp_s = HD_SWA * NKV_SWA;
        static std::vector<ggml_fp16_t> kf16, vf16, ks16, vs16;
        kf16.resize((size_t) kv_len * fpp_f); vf16.resize((size_t) kv_len * fpp_f);
        for (size_t i = 0; i < kf16.size(); ++i) { kf16[i] = ggml_fp32_to_fp16(acc_kf[i]); vf16[i] = ggml_fp32_to_fp16(acc_vf[i]); }
        ks16.resize((size_t) kv_swa * fpp_s); vs16.resize((size_t) kv_swa * fpp_s);
        const size_t off_s = (size_t)(kv_len - kv_swa) * fpp_s;
        for (size_t i = 0; i < ks16.size(); ++i) { ks16[i] = ggml_fp32_to_fp16(acc_ks[off_s + i]); vs16[i] = ggml_fp32_to_fp16(acc_vs[off_s + i]); }
        io.kv_len_full = kv_len; io.kv_len_swa = kv_swa; io.n_tokens = 1;
        io.embd = wide.data();
        io.k_full = kf16.data(); io.v_full = vf16.data();
        io.k_swa  = ks16.data(); io.v_swa  = vs16.data();
        llama_gemma4_assistant_set_io(dft, &io);

        llama_memory_clear(llama_get_memory(ctx_dft), true);
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1; b.token[0] = 0; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        if (llama_decode(ctx_dft, b) != 0) { fprintf(stderr, "draft decode failed\n"); exit(1); }
        const float * dl = llama_get_logits_ith(ctx_dft, 0);
        llama_token dtok = argmax(dl, n_vocab);
        const float * dn = llama_get_embeddings_ith(ctx_dft, 0); // post-norm hidden [1024]
        cur_h = post_proj(dn);
        llama_batch_free(b);
        return dtok;
    };

    // ---- speculative loop (greedy, single sequence; option-a incremental KV) ----
    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    std::vector<llama_token> seq = toks;
    std::vector<float> logits_last(n_vocab), hidden_last, plog, phid;
    decode_incr(seq, 0, plog, phid);                              // prompt: cache 0..P-1, capture full KV
    append_cap();                                                 // acc = prompt shared KV
    memcpy(logits_last.data(), plog.data() + (size_t)(seq.size() - 1) * n_vocab, n_vocab * sizeof(float));
    hidden_last = phid;

    if (getenv("G4A_DEBUG_KV")) {
        // extend 4 tokens incrementally (greedy), accumulating KV
        for (int i = 0; i < 4; ++i) {
            llama_token tt = argmax(logits_last.data(), n_vocab);
            std::vector<float> l2, h2;
            decode_incr(std::vector<llama_token>{tt}, (int) seq.size(), l2, h2);
            append_cap();
            memcpy(logits_last.data(), l2.data(), n_vocab * sizeof(float));
            hidden_last = h2; seq.push_back(tt);
        }
        std::vector<float> inc_kf = acc_kf, inc_ks = acc_ks; // incremental accumulation
        // full forward of the same sequence (cache empty)
        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        std::vector<float> lg, hd;
        decode_incr(seq, 0, lg, hd);
        double mxf = 0, mxs = 0;
        for (size_t i = 0; i < cap.k_full.size(); ++i) mxf = std::max(mxf, (double) fabs(cap.k_full[i] - inc_kf[i]));
        for (size_t i = 0; i < cap.k_swa.size();  ++i) mxs = std::max(mxs, (double) fabs(cap.k_swa[i]  - inc_ks[i]));
        printf("KV diff incremental-vs-full: k_full max=%g (n=%zu/%zu)  k_swa max=%g (n=%zu/%zu)\n",
               mxf, cap.k_full.size(), inc_kf.size(), mxs, cap.k_swa.size(), inc_ks.size());
        return 0;
    }

    std::vector<llama_token> spec_out;
    int n_acc_tot = 0, n_draft_tot = 0, cycles = 0;
    std::vector<int> raw_match(K, 0),  raw_total(K, 0);   // drafts[k] == target argmax[k] (ignores chain)
    std::vector<int> cond_match(K, 0), cond_total(K, 0);  // in-chain acceptance (all prior matched)
    while ((int) spec_out.size() < n_predict) {
        const int L = (int) seq.size();                          // == acc_len; cache holds 0..L-1
        // draft K tokens from fixed position L-1 over the accumulated backbone KV
        std::vector<llama_token> drafts;
        llama_token cur_tok = seq[L - 1];
        std::vector<float> cur_h = hidden_last;
        for (int k = 0; k < K; ++k) { llama_token dt = draft_step(cur_tok, cur_h, /*kv_len=*/ L, /*pos=*/ L - 1); drafts.push_back(dt); cur_tok = dt; }
        n_draft_tot += K;
        // verify: incremental decode of drafts at L..L+K-1 (cap discarded)
        std::vector<float> vlog, vhid;
        decode_incr(drafts, L, vlog, vhid);
        // targ[j] predicts position L+j: j==0 from logits_last (pos L-1), else from verify logits
        std::vector<llama_token> targ(K + 1);
        targ[0] = argmax(logits_last.data(), n_vocab);
        for (int j = 1; j <= K; ++j) targ[j] = argmax(vlog.data() + (size_t)(j - 1) * n_vocab, n_vocab);
        for (int j = 0; j < K; ++j) { raw_total[j]++; if (targ[j] == drafts[j]) raw_match[j]++; }
        // accept matched prefix + 1 bonus (all emitted are target argmax => lossless)
        std::vector<llama_token> accepted;
        bool chain = true;
        for (int j = 0; j <= K; ++j) {
            accepted.push_back(targ[j]);
            if (j < K && chain) { cond_total[j]++; if (targ[j] == drafts[j]) cond_match[j]++; else chain = false; }
            if (j == K || targ[j] != drafts[j]) break;
        }
        const int m = (int) accepted.size();
        n_acc_tot += (m - 1);
        // roll back the K verify positions, then commit the m accepted tokens (correct cache + KV)
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, L, -1);
        std::vector<float> clog, chid;
        decode_incr(accepted, L, clog, chid);                    // cache -> 0..L+m-1; cap = m new positions
        append_cap();                                            // acc grows by m
        memcpy(logits_last.data(), clog.data() + (size_t)(m - 1) * n_vocab, n_vocab * sizeof(float));
        hidden_last = chid;
        for (auto t : accepted) { seq.push_back(t); spec_out.push_back(t); }
        cycles++;
    }

    // ---- baseline: target-only greedy (incremental) ----
    std::vector<llama_token> base_out;
    {
        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        std::vector<float> lg, hd;
        decode_incr(toks, 0, lg, hd);
        std::vector<float> last(lg.end() - n_vocab, lg.end());
        for (int i = 0; i < (int) spec_out.size(); ++i) {
            llama_token tt = argmax(last.data(), n_vocab);
            base_out.push_back(tt);
            std::vector<float> l2, h2;
            decode_incr(std::vector<llama_token>{tt}, (int) toks.size() + i, l2, h2);
            last.assign(l2.end() - n_vocab, l2.end());
        }
    }

    // ---- report ----
    bool lossless = (spec_out == base_out);
    printf("\n=== spec_run ===\n");
    auto detok = [&](const std::vector<llama_token> & t) {
        std::string s; char buf[256];
        for (auto id : t) { int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true); if (n>0) s.append(buf, n); }
        return s;
    };
    printf("spec output: %s\n", detok(spec_out).c_str());
    printf("base output: %s\n", detok(base_out).c_str());
    printf("LOSSLESS: %s\n", lossless ? "YES" : "NO");
    printf("acceptance: %d/%d drafted tokens accepted over %d cycles (%.1f%%); ~%.2f tokens/cycle\n",
           n_acc_tot, n_draft_tot, cycles, 100.0 * n_acc_tot / (n_draft_tot ? n_draft_tot : 1),
           (double) spec_out.size() / (cycles ? cycles : 1));
    printf("per-k draft quality:\n");
    for (int k = 0; k < K; ++k) {
        printf("  k=%d  raw %2d/%-2d (%5.1f%%)   in-chain %2d/%-2d (%5.1f%%)\n", k,
               raw_match[k], raw_total[k], 100.0 * raw_match[k] / (raw_total[k] ? raw_total[k] : 1),
               cond_match[k], cond_total[k], 100.0 * cond_match[k] / (cond_total[k] ? cond_total[k] : 1));
    }

    llama_free(ctx_dft); llama_model_free(dft);
    llama_free(ctx_tgt); llama_model_free(tgt);
    llama_backend_free();
    return 0;
}
