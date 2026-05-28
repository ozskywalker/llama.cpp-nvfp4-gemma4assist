// Probe: capture the Gemma-4 backbone's shared-KV tensors via a cb_eval callback.
//
// Goal (driver step 1): confirm that the speculative driver can pull the backbone's
// per-layer-type shared K/V (and the final hidden state) out of a live target decode,
// and identify the exact tensor names / layers / shapes the draft needs.
//
// build:
//   g++ -O2 -std=c++17 devtools/gemma4_assistant/probe_kv.cpp -o /tmp/g4a_probe \
//       -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$PWD/build/bin
// run:
//   /tmp/g4a_probe /models/huggingface/google-gemma-4-31B_NVFP4.gguf 99

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct Cap {
    std::vector<std::string> log;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    Cap * c = (Cap *) ud;
    const char * nm = t->name;
    const bool want =
        strncmp(nm, "Kcur", 4) == 0 ||
        strncmp(nm, "Vcur", 4) == 0 ||
        strncmp(nm, "l_out", 5) == 0 ||
        strncmp(nm, "result_norm", 11) == 0;
    if (ask) {
        return want;            // request these nodes' data
    }
    if (want) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%-20s type=%-6s ne=[%lld, %lld, %lld, %lld]",
                 nm, ggml_type_name(t->type),
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        c->log.push_back(buf);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <backbone.gguf> [n_gpu_layers]\n", argv[0]); return 1; }
    const char * mpath = argv[1];
    const int ngl = argc > 2 ? atoi(argv[2]) : 99;

    llama_backend_init();
    Cap cap;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 256;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "The quick brown fox jumps over the lazy dog.";
    std::vector<llama_token> toks(64);
    int n = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks.data(), (int) toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);

    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n - 1);
    }

    printf("model: n_embd=%d n_layer=%d  decoding %d tokens...\n",
           llama_model_n_embd(model), llama_model_n_layer(model), n);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    printf("=== captured tensors (graph/compute order) ===\n");
    for (const auto & s : cap.log) {
        printf("  %s\n", s.c_str());
    }
    printf("(total %zu records)\n", cap.log.size());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
