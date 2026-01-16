/*use libloading::Library;

fn has_cuda() -> bool {
    #[cfg(target_os = "windows")]
    let lib_name = "nvcuda.dll";

    #[cfg(target_os = "linux")]
    let lib_name = "libcuda.so.1";

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    let lib_name = "libcuda.so";

    unsafe { Library::new(lib_name).is_ok() }
}*/

fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("clip.cpp")
        .file("rust_interface.cpp")
        .flag("-Wno-ignored-qualifiers")
        .flag("-Wno-sign-compare")
        .flag("-Wno-unused-variable")
        .flag("-march=native")  
        .flag("-mavx")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-mf16c")
        .flag("-O3")
        .flag("-ffast-math")
        .flag("-funroll-loops")
        .compile("clipcpp");
    
    let mut ggml = cc::Build::new();
    ggml.file("ggml/src/ggml.c")
        .flag("-std=c11")
        .flag("-Wno-unused-variable")
        .flag("-march=native")
        .flag("-mavx")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-mf16c")
        .flag("-O3")
        .flag("-ffast-math")
        .flag("-funroll-loops")
        .define("_GNU_SOURCE", None)
        .define("_POSIX_C_SOURCE", "200809L");

    /*if has_cuda() {
        ggml.flag("-DGGML_USE_CUBLAS").file("ggml-cuda.cu");
        println!("CUDA support available");
    }*/

    ggml.compile("ggml");

    println!("cargo:rerun-if-changed=clip.cpp");
    println!("cargo:rerun-if-changed=rust_interface.cpp");
    println!("cargo:rerun-if-changed=clip.h");
    println!("cargo:rerun-if-changed=ggml/src/ggml.c");
}
