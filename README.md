# my-nn-engine

ONNX model compiler with CPU and CUDA backends.

## Testing

```sh
cargo test-cpu                # CPU-only tests
cargo test                    # All tests (including CUDA)
```

### Saving build artifacts

```sh
MY_ONNX_SAVE_BUILD_DIR=1 RUST_LOG=info cargo test-cpu
```

This saves LLVM IR and transformed ONNX models to the build directory shown in the log output.

## Profiling with Nsight Systems

```sh
# Build the profile binary
cargo build --release --bin profile

# Profile ResNet18 (5 runs, 2 streams)
nsys profile target/release/profile models/validated/resnet18-v2-7 5 2

# Profile ResNet152 (10 runs, 1 stream)
nsys profile target/release/profile models/validated/resnet152-v2-7 10 1
```

Arguments: `<model_dir> [num_runs] [num_streams]`

## ORT Benchmark (via Podman)

Requires `nvidia-container-toolkit`.

```sh
bash bench/ort/run.sh cpu                       # CPU
bash bench/ort/run.sh cuda                      # CUDA
bash bench/ort/run.sh cpu --models resnet18-v2-7 --runs 10
```
