## Build

```bash
cargo build -p onnx-test-tools --release
```

## Usage

### Extraction

```bash
./target/release/onnx-test-tools \
  --mode extract \
  --model models/validated/MODEL/MODEL.onnx \
  --inputs models/validated/MODEL/test_data_set_0/input_*.pb \
  --outputs "output_node_name" \
  --output-dir models/extracted/MODEL/subgraph_name
```

### Binary Search

```bash
# Build the test helper
cargo build -p onnx-test-tools --example test_extracted --release

# Run binary search via container
./target/release/onnx-test-tools \
  --mode binary-search \
  --model models/validated/bertsquad-12/bertsquad-12.onnx \
  --inputs models/validated/bertsquad-12/test_data_set_0/input_*.pb \
  --test-command ./target/release/examples/test_extracted "{extract_dir}" 0.01
```
