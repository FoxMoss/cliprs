#pragma once

#include "rust/cxx.h"
#include <optional>
#include <string>
#include <vector>

void init(rust::String model_path);
rust::vec<float> embed_text(rust::String text);
rust::vec<float> embed_image(rust::String path);
float embed_compare(const rust::vec<float> & p1, const rust::vec<float> & p2);
void end();

extern "C" void clip_log_warning(const char* message);