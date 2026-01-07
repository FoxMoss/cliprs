use clipers::{rust_embed_compare, rust_embed_image, rust_embed_text, rust_end, rust_init};


fn main(){
    rust_init("/home/foxmoss/Downloads/clip-vit-large-patch14_ggml-model-q8_0.gguf");

    let text_embbedding = rust_embed_text("a tall man".to_string()).unwrap();
    print!("{:?}\n", text_embbedding);
    let image_embbedding = rust_embed_image("/home/foxmoss/Downloads/tallman.jpg".to_string()).unwrap();
    print!("{:?}\n", image_embbedding);

    let score = rust_embed_compare(&text_embbedding, &image_embbedding);
    println!("score {}", score);


    rust_end();
}
