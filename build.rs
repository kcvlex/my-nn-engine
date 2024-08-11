extern crate prost_build;

fn main() -> std::io::Result<()> {
    prost_build::compile_protos(&["third-party/onnx/onnx/onnx.proto3"], &["third-party/"])?;
    Ok(())
}
