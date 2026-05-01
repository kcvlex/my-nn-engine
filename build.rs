extern crate prost_build;

fn main() -> std::io::Result<()> {
    prost_build::compile_protos(&["third-party/onnx/onnx/onnx.proto3"], &["third-party/"])?;

    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let cuda_path = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| {
                for p in ["/opt/cuda", "/usr/local/cuda"] {
                    if std::path::Path::new(p).exists() {
                        return p.to_string();
                    }
                }
                "/usr/local/cuda".to_string()
            });
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
    }
    Ok(())
}
