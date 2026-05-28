// End-to-end gate for build_arch_graph through the real llama API.
//
// Loads the draft GGUF, attaches the oracle inputs via llama_gemma4_assistant_set_io,
// runs llama_decode, and dumps logits + the post-norm hidden (embeddings output) so
// compare_outputs.py can diff them against hf_ref.npz. Runs on CPU (-ngl 0).
//
// build:
//   g++ -O2 -std=c++17 devtools/gemma4_assistant/test_decode.cpp -o /tmp/g4a_decode \
//       -Iinclude -Isrc -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$PWD/build/bin

#include "llama.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

static std::vector<float> read_f32(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(n / 4);
    if (fread(v.data(), 1, n, f) != (size_t) n) { exit(1); }
    fclose(f); return v;
}
static std::vector<int32_t> read_i32(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(n / 4);
    if (fread(v.data(), 1, n, f) != (size_t) n) { exit(1); }
    fclose(f); return v;
}
static void write_bin(const std::string & p, const void * d, size_t n) {
    FILE * f = fopen(p.c_str(), "wb"); fwrite(d, 1, n, f); fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s <draft.gguf> <in_dir> <out_dir> <kv_len> <q_len> [n_gpu_layers]\n", argv[0]); return 1; }
    const std::string mpath = argv[1], indir = argv[2], outdir = argv[3];
    const int kv_len = atoi(argv[4]);
    const int q      = atoi(argv[5]);
    const int ngl    = argc > 6 ? atoi(argv[6]) : 0; // 0 = CPU; 99 = offload all to GPU

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(mpath.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = 64;
    cp.n_batch      = 64;
    cp.n_ubatch     = 64;
    cp.embeddings   = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    auto embd = read_f32(indir + "/inp_embd.bin");
    auto kf = read_f32(indir + "/k_full.bin"), vf = read_f32(indir + "/v_full.bin");
    auto ks = read_f32(indir + "/k_swa.bin"),  vs = read_f32(indir + "/v_swa.bin");
    auto pos = read_i32(indir + "/pos.bin");

    llama_gemma4_assistant_io io{};
    io.kv_len = kv_len; io.n_tokens = q;
    io.embd = embd.data();
    io.k_full = kf.data(); io.v_full = vf.data();
    io.k_swa = ks.data();  io.v_swa = vs.data();
    llama_gemma4_assistant_set_io(model, &io);

    llama_batch batch = llama_batch_init(q, 0, 1);
    batch.n_tokens = q;
    for (int i = 0; i < q; ++i) {
        batch.token[i]    = 0;          // unused (graph uses io->embd)
        batch.pos[i]      = pos[i];
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == q - 1);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_embd  = llama_model_n_embd(model);
    float * logits = llama_get_logits_ith(ctx, q - 1);
    float * emb    = llama_get_embeddings_ith(ctx, q - 1);
    write_bin(outdir + "/logits.bin", logits, (size_t) n_vocab * sizeof(float));
    write_bin(outdir + "/model_norm_out.bin", emb, (size_t) n_embd * sizeof(float));

    int am = 0; for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[am]) am = i;
    printf("decode ok: argmax(logits)=%d  (expected HF=17887 for the default oracle)\n", am);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
