# Cliprs

[clip.cpp](https://github.com/monatis/clip.cpp) bindings to rust.

## Grabing a model file

I've been using
[huggingface.co/mys/ggml_clip-vit-large-patch14](https://huggingface.co/mys/ggml_clip-vit-large-patch14)
two tower models for testing. All quantization levels have worked great!

## Install

To add to your project:

```
cargo add cliprs
```

## Usage

Check [simon0302010/findimg/](https://github.com/simon0302010/findimg/) for a more detailed
example.

But basic usage looks like
```rust
use cliprs::ClipModel;

fn main() {
    let model = ClipModel::new("/home/simon/Dokumente/Rust/findimg/clip-vit-large-patch14_ggml-model-q8_0.gguf");

    let text_embedding = model.embed_text("party").unwrap();

    let image_embedding = model.embed_image("/home/simon/Downloads/party.jpg").unwrap();

    let similarity = model.embed_compare(&text_embedding, &image_embedding);
    println!("Similarity: {}", similarity);

    for warning in cliprs::poll_warnings() {
        eprintln!("Warning: {}", warning);
    }
}
```
