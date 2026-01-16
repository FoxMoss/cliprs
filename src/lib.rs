use std::sync::Mutex;

static WARNINGS: Mutex<Vec<String>> = Mutex::new(Vec::new());
const SUPPORTED_IMAGE_FORMATS: [&str; 10] = ["jpg", "jpeg", "png", "tga", "bmp", "psd"," gif", "hdr", "pic", "ppm"];

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        fn log_warning(message: String);
    }

    unsafe extern "C++" {
        include!("cliprs/clip.h");
        include!("cliprs/rust_interface.h");

        type clip_ctx;

        fn init(path: String) -> *mut clip_ctx;
        unsafe fn embed_text(ctx: *const clip_ctx, text: String) -> Result<Vec<f32>>;
        unsafe fn embed_image(ctx: *const clip_ctx, path: String) -> Result<Vec<f32>>;
        unsafe fn embed_compare(ctx: *const clip_ctx, p1: &Vec<f32>, p2: &Vec<f32>) -> f32;
        unsafe fn end(ctx: *mut clip_ctx);
    }
}

/// Is being called by the C++ side to log warnings.
/// Although not intended, it could still be used to log warnings that can be read using the `poll_warnings()` function.
pub fn log_warning(message: String) {
    if let Ok(mut warnings) = WARNINGS.lock() {
        warnings.push(message);
    }
}

/// Represents a CLIP model loaded from a GGUF file.
/// 
/// # Example
/// ```no_run
/// let model = cliprs::ClipModel::new("path/to/model.gguf");
/// 
/// let text_embedding = model.embed_text("A sample text").unwrap();
/// 
/// let image_embedding = model.embed_image("path/to/image.png").unwrap();
/// 
/// let similarity = model.embed_compare(&text_embedding, &image_embedding);
/// 
/// println!("Similarity: {}", similarity);
/// ```
pub struct ClipModel {
    ctx: *mut ffi::clip_ctx,
}

// Required for Send/Sync since we're using raw pointers
unsafe impl Send for ClipModel {}
unsafe impl Sync for ClipModel {}

impl ClipModel {
    /// Initializes ClipModel with a model path.
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// let model = cliprs::ClipModel::new("path/to/model.gguf");
    /// ```
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            ctx: ffi::init(model_path.into()),
        }
    }

    /// Compares two embeddings and returns the similarity.
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// let similarity = model.embed_compare(&text_embedding, &image_embedding);
    /// ```
    pub fn embed_compare(&self, p1: &Vec<f32>, p2: &Vec<f32>) -> f32 {
        unsafe { ffi::embed_compare(self.ctx, p1, p2) }
    }

    /// Creates an embedding for a string
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// let text_embedding = model.embed_text("blue car");
    /// ```
    pub fn embed_text(&self, text: impl Into<String>) -> Result<Vec<f32>, String> {
        match unsafe { ffi::embed_text(self.ctx, text.into()) } {
            Ok(embed) if embed.is_empty() => Err("Text embedding is empty".to_string()),
            Ok(embed) => Ok(embed),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Creates an image embedding from a path
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// let image_embedding = model.embed_image("path/to/image.jpg")
    /// ```
    pub fn embed_image(&self, path: impl Into<String>) -> Result<Vec<f32>, String> {
        let path: String = path.into();

        if !SUPPORTED_IMAGE_FORMATS.iter().any(|suffix| path.ends_with(suffix)) {
            return Err(format!("Unsupported image format: {}", path.split('.').last().unwrap_or_default()));
        }

        match unsafe { ffi::embed_image(self.ctx, path.clone()) } {
            Ok(embed) if embed.is_empty() => Err(format!("Embedding for {} is empty", path)),
            Ok(embed) => Ok(embed),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Drop for ClipModel {
    fn drop(&mut self) {
        unsafe {
            ffi::end(self.ctx);
        }
    }
}

/// Returns all warnings emitted by all Clip instances since the last call
/// 
/// # Example
/// 
/// ```
/// let warnings = cliprs::poll_warnings();
/// println!("{:?}", warnings);
/// ```
pub fn poll_warnings() -> Vec<String> {
    WARNINGS
        .lock()
        .map(|mut w| w.drain(..).collect())
        .unwrap_or_default()
}