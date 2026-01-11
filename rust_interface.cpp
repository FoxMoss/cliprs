#include "clip.h"
#include <cstdint>
#include <optional>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cliprs/src/lib.rs.h"
#include "rust_interface.h"

// for clip.cpp to call
extern "C" void clip_log_warning(const char* message) {
    log_warning(rust::String(message));
}

static struct clip_ctx * ctx;
const int verbosity = 1;
static int n_threads = 1;
static int vec_dim;
void init(rust::String model_path) {
    n_threads = std::thread::hardware_concurrency();
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
        throw std::runtime_error(std::string(__func__) + "Failed to encode text");
    }

    rust::vec<float> ret;
    std::copy(txt_vec.begin(), txt_vec.end(), std::back_inserter(ret));

    return ret;
}

rust::vec<float> embed_image(rust::String path) {
    std::string path_str(path);

    struct clip_image_u8 * img0 = clip_image_u8_make();
    if (!clip_image_load_from_file(path.c_str(), img0)) {
        throw std::runtime_error(std::string(__func__) + ": Failed to load image from " + path_str);
    }

    struct clip_image_f32 * img_res = clip_image_f32_make();
    if (!clip_image_preprocess(ctx, img0, img_res)) {
        throw std::runtime_error(std::string(__func__) + ": Failed to preprocess " + path_str);
    }

    std::vector<float> img_vec;
    img_vec.insert(img_vec.end(), vec_dim, 0);
    if (!clip_image_encode(ctx, n_threads, img_res, img_vec.data(), true)) {
        throw std::runtime_error(std::string(__func__) + ": Failed to encode " + path_str);
    }

    rust::vec<float> ret;
    std::copy(img_vec.begin(), img_vec.end(), std::back_inserter(ret));
    return ret;
}

float embed_compare(const rust::vec<float> & p1, const rust::vec<float> & p2) {
    return clip_similarity_score(p1.data(), p2.data(), vec_dim);
}

void end() { clip_free(ctx); }
