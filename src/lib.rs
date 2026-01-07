use crate::ffi::{init, end, embed_text, embed_image, embed_compare};

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("clipers/clip.h");
        include!("clipers/rust_interface.h");

        fn init(path: String);
        fn embed_text(text: String) -> Vec<f32>;
        fn embed_image(path: String) -> Vec<f32>;
        fn embed_compare(p1: &Vec<f32>, p2: &Vec<f32>) -> f32;
        fn end();
    }
}

pub fn rust_init(model_path: &str) {
    init(model_path.to_string());
}

pub fn rust_end() {
    end();
}

pub fn rust_embed_text(text: String) -> Option<Vec<f32>>{
    let vec = embed_text(text);

    if vec.len() == 0{
        return None;
    }
    return Some(vec);
}

pub fn rust_embed_compare(p1: &Vec<f32>, p2: &Vec<f32>) -> f32{
    embed_compare(p1, p2)
}

pub fn rust_embed_image(path: String) -> Option<Vec<f32>>{
    let vec = embed_image(path);

    if vec.len() == 0{
        return None;
    }
    return Some(vec);
}


