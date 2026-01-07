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
    
    cc::Build::new()
        .file("ggml/src/ggml.c")
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
        // .flag("-DGGML_USE_OPENBLAS")
        .define("_GNU_SOURCE", None) 
        .define("_POSIX_C_SOURCE", "200809L")  
        .compile("ggml");
    
    println!("cargo:rerun-if-changed=clip.cpp");
    println!("cargo:rerun-if-changed=rust_interface.cpp");
    println!("cargo:rerun-if-changed=clip.h");
    println!("cargo:rerun-if-changed=ggml/src/ggml.c");
    println!("cargo:rustc-link-lib=cblas");
}
