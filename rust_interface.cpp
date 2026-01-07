#include "rust_interface.h"
#include "clip.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

static struct clip_ctx * ctx;
const int verbosity = 1;
const int n_threads = 4;
static int vec_dim;
void init(rust::String model_path) {
    // char * model_path = "/home/foxmoss/Downloads/clip-vit-large-patch14_ggml-model-q8_0.gguf";

    ctx = clip_model_load(model_path.data(), verbosity);
    vec_dim = clip_get_vision_hparams(ctx)->projection_dim;
}

rust::vec<float> embed_text(rust::String text) {
    struct clip_tokens tokens;
    clip_tokenize(ctx, text.c_str(), &tokens);

    std::vector<float> txt_vec;
    txt_vec.insert(txt_vec.end(), vec_dim, 0);

    if (!clip_text_encode(ctx, n_threads, &tokens, txt_vec.data(), true)) {
        fprintf(stderr, "%s: failed to encode text\n", __func__);
        txt_vec.clear();
    }

    rust::vec<float> ret;
    std::copy(txt_vec.begin(), txt_vec.end(), std::back_inserter(ret));

    return ret;
}

rust::vec<float> embed_image(rust::String path) {
    struct clip_image_u8 * img0 = clip_image_u8_make();
    if (!clip_image_load_from_file(path.c_str(), img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, path.c_str());
        return {};
    }

    struct clip_image_f32 * img_res = clip_image_f32_make();
    if (!clip_image_preprocess(ctx, img0, img_res)) {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return {};
    }

    std::vector<float> img_vec;
    img_vec.insert(img_vec.end(), vec_dim, 0);
    if (!clip_image_encode(ctx, n_threads, img_res, img_vec.data(), true)) {
        fprintf(stderr, "%s: failed to encode image\n", __func__);
        return {};
    }

    rust::vec<float> ret;
    std::copy(img_vec.begin(), img_vec.end(), std::back_inserter(ret));
    return ret;
}

float embed_compare(const rust::vec<float> & p1, const rust::vec<float> & p2) {
    return clip_similarity_score(p1.data(), p2.data(), vec_dim);
}

void end() { clip_free(ctx); }
