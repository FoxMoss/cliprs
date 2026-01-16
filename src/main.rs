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