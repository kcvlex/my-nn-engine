# my-onnx

ONNX model compiler with CPU and CUDA backends.

## Testing

```sh
cargo test-cpu                # CPU-only tests
cargo test                    # All tests (including CUDA)
cargo bench-cpu               # CPU benchmarks
cargo bench-cuda              # CUDA benchmarks
```

### Saving build artifacts

```sh
MY_ONNX_SAVE_BUILD_DIR=1 RUST_LOG=info cargo test-cpu
```

This saves LLVM IR and transformed ONNX models to the build directory shown in the log output.
