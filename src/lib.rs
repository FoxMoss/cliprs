use crate::ffi::{embed_compare, embed_image, embed_text, end, init};

use std::sync::Mutex;

static WARNINGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        fn log_warning(message: String);
    }

    unsafe extern "C++" {
        include!("cliprs/clip.h");
        include!("cliprs/rust_interface.h");

        fn init(path: String);
        fn embed_text(text: String) -> Result<Vec<f32>>;
        fn embed_image(path: String) -> Result<Vec<f32>>;
        fn embed_compare(p1: &Vec<f32>, p2: &Vec<f32>) -> f32;
        fn end();
    }
}

pub fn cliprs_init(model_path: &str) {
    init(model_path.to_string());
}

pub fn cliprs_end() {
    end();
}

pub fn cliprs_embed_compare(p1: &Vec<f32>, p2: &Vec<f32>) -> f32{
    embed_compare(p1, p2)
}

/// Embeds text. Will return Err(e) if it fails. Non-fatal warnings are accesible using `poll_warnings()`
pub fn cliprs_embed_text(text: impl Into<String>) -> Result<Vec<f32>, String> {
    match embed_text(text.into()) {
        Ok(embed) => if embed.is_empty() {
            Err("Text embedding is empty".to_string())
        } else {
            Ok(embed)
        },
        Err(e) => Err(e.to_string())
    }
}

/// Embeds am image from a path. Will return Err(e) if it fails. Non-fatal warnings are accesible using `poll_warnings()`
pub fn cliprs_embed_image(path: impl Into<String>) -> Result<Vec<f32>, String> {
    let path: String = path.into();
    match embed_image(path.clone()) {
        Ok(embed) => if embed.is_empty() {
            Err(format!("Embedding for {} is empty", path))
        } else {
            Ok(embed)
        },
        Err(e) => Err(e.to_string())
    }
}

fn log_warning(message: String) {
    if let Ok(mut warnings) = WARNINGS.lock() {
        warnings.push(message);
    }
}

pub fn poll_warnings() -> Vec<String> {
    WARNINGS.lock()
        .map(|mut w| w.drain(..).collect())
        .unwrap_or_default()
}