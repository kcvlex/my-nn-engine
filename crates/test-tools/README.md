# my-nn-engine-test-tools

A collection of tools primarily intended for debugging.

## Build

```bash
cargo build -p my-nn-engine-test-tools --release
```

## Usage

### Extraction

Extract a subgraph from an ONNX model whose last node is the specified one.

```bash
./target/release/my-nn-engine-test-tools \
  --mode extract \
  --model-path models/validated/MODEL/MODEL.onnx \
  --input-paths models/validated/MODEL/test_data_set_0/input_*.pb \
  --output-node-name "output_node_name" \
  --output-dir models/extracted/MODEL/subgraph_name
```

### Binary Search

Run a binary search to find the node responsible for a test failure.
Reference outputs are generated via onnxruntime for the extracted subgraph and compared against the test command's output.

```bash
# Build the test helper
cargo build -p my-nn-engine-test-tools --example test_extracted --release

# Run binary search
./target/release/my-nn-engine-test-tools \
  --mode binary-search \
  --model-path models/validated/bertsquad-12/bertsquad-12.onnx \
  --input-paths models/validated/bertsquad-12/test_data_set_0/input_*.pb \
  --test-command ./target/release/examples/test_extracted \
  --output-dir tmp/binary_search_nodes
```

The test command receives the extracted model directory via the `EXTRACTED_MODEL_DIR` environment variable.
Comparison epsilon can be configured via the `EPSILON` environment variable (default: 0.01).
