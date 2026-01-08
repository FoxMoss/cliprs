use cliprs::{cliprs_embed_compare, cliprs_embed_image, cliprs_embed_text, cliprs_end, cliprs_init};

fn main() {
    cliprs_init("/home/foxmoss/Downloads/clip-vit-large-patch14_ggml-model-q8_0.gguf");

    let text_embbedding = cliprs_embed_text("a tall man".to_string()).unwrap();
    print!("{:?}\n", text_embbedding);
    let image_embbedding =
        cliprs_embed_image("/home/foxmoss/Downloads/tallman.jpg".to_string()).unwrap();
    print!("{:?}\n", image_embbedding);

    let score = cliprs_embed_compare(&text_embbedding, &image_embbedding);
    println!("score {}", score);

    cliprs_end();
}
