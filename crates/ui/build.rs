fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().compile_protos(
        &[
            "proto/onnx_service.proto",
            "../../third-party/onnx/onnx/onnx.proto3",
        ],
        &["proto/", "../../third-party/onnx/onnx/"],
    )?;
    Ok(())
}
