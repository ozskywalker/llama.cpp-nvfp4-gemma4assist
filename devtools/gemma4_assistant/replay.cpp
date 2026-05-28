// Standalone ggml replay harness for the Gemma 4 Assistant draft forward.
//
// Loads the draft GGUF weights, reads the oracle's inputs (raw .bin from
// export_inputs.py), rebuilds the forward with raw ggml ops following the
// numpy-verified spec, computes on CPU, and writes outputs (.bin) for
// compare_outputs.py to diff against hf_ref.npz. This gates the ggml-level math
// (rope/attention/layout) against ground truth, independently of llama_context.
//
// build (from repo root, after building libggml):
//   g++ -O2 -std=c++17 devtools/gemma4_assistant/replay.cpp -o /tmp/g4a_replay \
//       -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,build/bin
//   (linking libllama pulls in the ggml libs; -lggml alone also works)

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// fixed config for google/gemma-4-31B-it-assistant
static const int   N_LAYER   = 4;
static const int   N_EMBD    = 1024;
static const int   N_HEAD    = 32;
static const int   N_FF      = 8192;
static const int64_t N_VOCAB = 262144;
static const int   N_EMBD_BB = 5376;
static const int   HD_FULL   = 512;
static const int   HD_SWA    = 256;
static const int   NKV_FULL  = 4;
static const int   NKV_SWA   = 16;
static const float EPS       = 1e-6f;
static const float THETA_FULL = 1000000.0f;
static const float THETA_SWA  = 10000.0f;
static const bool  SWA[N_LAYER] = {true, true, true, false};

static std::vector<char> read_file(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> b(n);
    if (fread(b.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return b;
}

static void write_bin(const std::string & p, const void * data, size_t n) {
    FILE * f = fopen(p.c_str(), "wb");
    fwrite(data, 1, n, f);
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <weights.gguf> <inputs_dir> <out_dir> <kv_len> [q_len]\n", argv[0]);
        return 1;
    }
    const std::string wpath = argv[1], indir = argv[2], outdir = argv[3];
    const int kv_len = atoi(argv[4]);
    const int q      = argc > 5 ? atoi(argv[5]) : 1;

    // --- load weights ---
    struct ggml_context * wctx = nullptr;
    struct gguf_init_params gp = { /*no_alloc=*/ false, /*ctx=*/ &wctx };
    struct gguf_context * gguf = gguf_init_from_file(wpath.c_str(), gp);
    if (!gguf) { fprintf(stderr, "failed to load %s\n", wpath.c_str()); return 1; }

    auto W = [&](const std::string & n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n.c_str());
        if (!t) { fprintf(stderr, "missing weight %s\n", n.c_str()); exit(1); }
        return t;
    };
    auto Wl = [&](int il, const std::string & suf) {
        char buf[128]; snprintf(buf, sizeof(buf), "blk.%d.%s", il, suf.c_str());
        return W(buf);
    };

    // --- compute context (metadata only; gallocr allocates data) ---
    size_t meta = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    struct ggml_init_params cp = { meta, nullptr, /*no_alloc=*/ true };
    struct ggml_context * ctx = ggml_init(cp);

    // input tensors (ne = reversed numpy shape from export_inputs.py)
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * N_EMBD_BB, q);   ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);                  ggml_set_input(pos);
    ggml_tensor * kf  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD_FULL, NKV_FULL, kv_len); ggml_set_input(kf);
    ggml_tensor * vf  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD_FULL, NKV_FULL, kv_len); ggml_set_input(vf);
    ggml_tensor * ks  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD_SWA, NKV_SWA, kv_len);   ggml_set_input(ks);
    ggml_tensor * vs  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD_SWA, NKV_SWA, kv_len);   ggml_set_input(vs);

    auto rms = [&](ggml_tensor * x, ggml_tensor * w) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, EPS), w);
    };

    // x = pre_projection @ inp   -> [n_embd, q]
    ggml_tensor * x = ggml_mul_mat(ctx, W("mtp.pre_projection.weight"), inp);

    std::vector<ggml_tensor *> layer_out(N_LAYER, nullptr);
    for (int il = 0; il < N_LAYER; ++il) {
        const bool swa = SWA[il];
        const int  hd   = swa ? HD_SWA  : HD_FULL;
        const int  nkv  = swa ? NKV_SWA : NKV_FULL;
        const float theta = swa ? THETA_SWA : THETA_FULL;
        ggml_tensor * K = swa ? ks : kf;
        ggml_tensor * V = swa ? vs : vf;
        ggml_tensor * ff_factors = swa ? nullptr : W("rope_freqs.weight");

        ggml_tensor * res = x;
        ggml_tensor * h = rms(x, Wl(il, "attn_norm.weight"));

        // Q: proj -> [hd, n_head, q] -> q_norm -> rope (NEOX)
        ggml_tensor * Q = ggml_mul_mat(ctx, Wl(il, "attn_q.weight"), h);     // [n_head*hd, q]
        Q = ggml_reshape_3d(ctx, Q, hd, N_HEAD, q);
        Q = rms(Q, Wl(il, "attn_q_norm.weight"));
        Q = ggml_rope_ext(ctx, Q, pos, ff_factors, hd, GGML_ROPE_TYPE_NEOX,
                          262144, theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // cross-attention over external K/V (build_attn_mha layout), scale 1.0, no mask
        ggml_tensor * q_p = ggml_permute(ctx, Q, 0, 2, 1, 3);               // [hd, q, n_head]
        ggml_tensor * k_p = ggml_permute(ctx, K, 0, 2, 1, 3);               // [hd, kvlen, nkv]
        ggml_tensor * kq  = ggml_mul_mat(ctx, k_p, q_p);                    // [kvlen, q, n_head] (GQA bcast)
        kq = ggml_soft_max(ctx, kq);
        ggml_tensor * v_p = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3)); // [kvlen, hd, nkv]
        ggml_tensor * kqv = ggml_mul_mat(ctx, v_p, kq);                    // [hd, q, n_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                          // [hd, n_head, q]
        ggml_tensor * attn = ggml_cont_2d(ctx, kqv, hd * N_HEAD, q);
        ggml_tensor * o = ggml_mul_mat(ctx, Wl(il, "attn_output.weight"), attn); // [n_embd, q]

        ggml_tensor * pa = rms(o, Wl(il, "post_attention_norm.weight"));
        ggml_tensor * sa = ggml_add(ctx, res, pa);

        ggml_tensor * f = rms(sa, Wl(il, "ffn_norm.weight"));
        ggml_tensor * gate = ggml_gelu(ctx, ggml_mul_mat(ctx, Wl(il, "ffn_gate.weight"), f));
        ggml_tensor * up   = ggml_mul_mat(ctx, Wl(il, "ffn_up.weight"), f);
        ggml_tensor * ffo  = ggml_mul_mat(ctx, Wl(il, "ffn_down.weight"), ggml_mul(ctx, gate, up));
        ffo = rms(ffo, Wl(il, "post_ffw_norm.weight"));
        x = ggml_add(ctx, sa, ffo);
        x = ggml_mul(ctx, x, Wl(il, "layer_output_scale.weight"));         // layer_scalar at end
        ggml_set_output(x);
        layer_out[il] = x;
    }

    ggml_tensor * nrm = rms(x, W("output_norm.weight"));
    ggml_tensor * logits = ggml_mul_mat(ctx, W("token_embd.weight"), nrm);  // [vocab, q]
    ggml_tensor * last_h = ggml_mul_mat(ctx, W("mtp.post_projection.weight"), nrm); // [n_embd_bb, q]
    ggml_set_output(logits);
    ggml_set_output(last_h);

    // --- build + allocate + compute on CPU ---
    ggml_backend_t backend = ggml_backend_cpu_init();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, last_h);
    for (auto * t : layer_out) ggml_build_forward_expand(gf, t);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    auto set_in = [&](ggml_tensor * t, const std::string & name) {
        std::vector<char> b = read_file(indir + "/" + name + ".bin");
        if (b.size() != ggml_nbytes(t)) {
            fprintf(stderr, "size mismatch %s: file %zu vs tensor %zu\n", name.c_str(), b.size(), ggml_nbytes(t));
            exit(1);
        }
        ggml_backend_tensor_set(t, b.data(), 0, b.size());
    };
    set_in(inp, "inp_embd"); set_in(pos, "pos");
    set_in(kf, "k_full"); set_in(vf, "v_full"); set_in(ks, "k_swa"); set_in(vs, "v_swa");

    ggml_backend_graph_compute(backend, gf);

    // --- dump outputs ---
    auto dump = [&](ggml_tensor * t, const std::string & name) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        write_bin(outdir + "/" + name + ".bin", buf.data(), buf.size() * sizeof(float));
    };
    for (int il = 0; il < N_LAYER; ++il) dump(layer_out[il], "layer." + std::to_string(il) + ".out");
    dump(logits, "logits");
    dump(last_h, "last_hidden_state");
    printf("replay done: wrote outputs to %s/\n", outdir.c_str());
    return 0;
}
