## Build

```bash
cargo build -p my-onnx-test-tools --release
```

## Usage

### Extraction

```bash
./target/release/my-onnx-test-tools \
  --mode extract \
  --model-path models/validated/MODEL/MODEL.onnx \
  --input-paths models/validated/MODEL/test_data_set_0/input_*.pb \
  --output-node-name "output_node_name" \
  --output-dir models/extracted/MODEL/subgraph_name
```

### Binary Search

Binary search runs the ONNX extraction and onnxruntime inference inside a container, but executes the test command on the host. This way the test binary can link against host libraries (e.g. `libLLVM.so`).

```bash
# Build the test helper
cargo build -p my-onnx-test-tools --example test_extracted --release

# Run binary search
./target/release/my-onnx-test-tools \
  --mode binary-search \
  --model-path models/validated/bertsquad-12/bertsquad-12.onnx \
  --input-paths models/validated/bertsquad-12/test_data_set_0/input_*.pb \
  --test-command ./target/release/examples/test_extracted \
  --output-dir tmp/binary_search_nodes
```

The test command receives the extracted model directory via the `EXTRACTED_MODEL_DIR` environment variable.
Comparison epsilon can be configured via the `EPSILON` environment variable (default: 0.01).
