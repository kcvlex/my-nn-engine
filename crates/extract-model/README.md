# ONNX Model Extraction Tool

This tool extracts a subgraph from an ONNX model, runs it to generate expected outputs, and saves everything in the format expected by the test suite.

## Why?

When debugging large ONNX models, it's helpful to extract smaller subgraphs to isolate issues. This tool:

1. Extracts a portion of the model from specified inputs to outputs
2. Runs the extracted model with the original input data using ONNX Runtime
3. Saves the model and test data in the format expected by `tests/extracted_models.rs`

## Prerequisites

### Option 1: Using Rust Binary with Docker (Recommended)
- Rust toolchain (to build the tool)
- Docker (no Python environment setup needed on the host!)

Build the tool once:
```bash
cargo build -p extract-model --release
```

### Option 2: Using Python Directly
If you prefer not to use Docker:
```bash
pip install -r requirements.txt
```

## Usage

### Using Rust Binary with Docker (Recommended)

```bash
# Build once (or use cargo run -p extract-model -- to skip building)
cargo build -p extract-model --release

# Run extraction
./target/release/extract-model \
  --model models/validated/MODEL/MODEL.onnx \
  --inputs models/validated/MODEL/test_data_set_0/input_*.pb \
  --outputs "output_node_name" \
  --output-dir models/extracted/MODEL/subgraph_name

# Or use cargo run directly (slower, but no need to build first)
cargo run -p extract-model -- \
  --model models/validated/MODEL/MODEL.onnx \
  --inputs models/validated/MODEL/test_data_set_0/input_*.pb \
  --outputs "output_node_name" \
  --output-dir models/extracted/MODEL/subgraph_name
```

### Using Python Directly

If you have Python dependencies installed:

```bash
cd /path/to/my-onnx  # Project root
python tools/extract-model/extract_model.py \
  --model models/validated/MODEL/MODEL.onnx \
  --inputs models/validated/MODEL/test_data_set_0/input_*.pb \
  --outputs "output_node_name" \
  --output-dir models/extracted/MODEL/subgraph_name
```

### Examples

#### Extract YOLOv4 subgraph

```bash
cargo run -p extract-model -- \
  --model models/validated/yolov4/yolov4.onnx \
  --inputs models/validated/yolov4/test_data_set_0/input_0.pb \
  --outputs "lambda_5/add:0" \
  --output-dir models/extracted/yolov4/until_lambda_5_add
```

#### Extract BERT subgraph with multiple inputs

```bash
cargo run -p extract-model -- \
  --model models/validated/bertsquad-12/bertsquad-12.onnx \
  --inputs models/validated/bertsquad-12/test_data_set_0/input_*.pb \
  --outputs "bert/encoder/Cast:0" \
  --output-dir models/extracted/bertsquad-12/until_encoder_cast
```

#### Extract with multiple output nodes

```bash
cargo run -p extract-model -- \
  --model models/validated/mymodel/mymodel.onnx \
  --inputs models/validated/mymodel/test_data_set_0/input_0.pb \
  --outputs "intermediate_node_1" "intermediate_node_2" \
  --output-dir models/extracted/mymodel/multi_output
```

## Finding Output Node Names

To find the names of nodes in your ONNX model, you can:

1. **Use Netron** (recommended): Open the model in [Netron](https://netron.app/) and click on nodes to see their names

2. **Use Python**:
   ```python
   import onnx
   model = onnx.load("model.onnx")
   for node in model.graph.node:
       print(f"{node.op_type}: {node.output}")
   ```

3. **Use grep on model file** (quick but crude):
   ```bash
   strings model.onnx | grep -i "node_name_part"
   ```

## Output Structure

The tool creates a directory with:

```
models/extracted/MODEL/subgraph_name/
├── model.onnx       # Extracted subgraph
├── input_0.pb       # Input tensor 0
├── input_1.pb       # Input tensor 1 (if multiple inputs)
└── output_0.pb      # Expected output tensor 0
```

## Using Extracted Models in Tests

Add a test in `tests/extracted_models.rs`:

```rust
#[test]
fn test_mymodel_subgraph() -> Result {
    run_test("mymodel/subgraph_name", 1e-2, Target::CPU, (1, 1))
}
```

Where `(1, 1)` is `(num_inputs, num_outputs)`.

## Options

- `--model PATH`: Path to the input ONNX model (relative to project root)
- `--inputs PATH...`: Paths to input tensor .pb files (can use wildcards like `input_*.pb`)
- `--outputs NAME...`: Names of output nodes to extract up to (can specify multiple)
- `--output-dir DIR`: Directory to save extracted model and test data
- `--input-names NAME...`: Override input node names (if needed)
- `--no-check`: Skip ONNX model validation (use if extraction fails with check errors)
- `--rebuild`: Rebuild the Docker image before running

## Troubleshooting

### "Error: Expected N input tensors but got M"

The tool auto-detects input names from the model. If you're starting extraction from intermediate nodes, specify input names explicitly:

```bash
./extract.sh \
  --model models/validated/MODEL/MODEL.onnx \
  --inputs input.pb \
  --outputs "output_node" \
  --output-dir models/extracted/MODEL/subgraph \
  --input-names "intermediate_node_name"
```

### "Model check failed"

If ONNX model validation fails but you know the extraction should work, use `--no-check`:

```bash
./extract.sh --no-check \
  --model ... \
  --inputs ... \
  --outputs ... \
  --output-dir ...
```

### Docker permission issues

The `extract.sh` script automatically detects if Docker requires sudo and uses it when needed. If you prefer to avoid sudo, add your user to the docker group:

```bash
sudo usermod -aG docker $USER
# Log out and log back in for changes to take effect
```

## Workflow for Debugging Large Models

1. **Run the full model test** and note which operation fails
2. **Find a node just before the failure** using Netron or by examining the model
3. **Extract up to that node**:
   ```bash
   ./extract.sh \
     --model models/validated/MODEL/MODEL.onnx \
     --inputs models/validated/MODEL/test_data_set_0/input_*.pb \
     --outputs "node_before_failure" \
     --output-dir models/extracted/MODEL/debug_1
   ```
4. **Add a test** in `tests/extracted_models.rs`
5. **Run the test** - if it passes, extract further; if it fails, you've isolated the issue
6. **Repeat** until you find the minimal failing case

## Development

To modify the extraction script:

1. Edit `extract_model.py`
2. Rebuild the Docker image: `./extract.sh --rebuild --help`
3. Test your changes

## Technical Details

- Uses `onnx.utils.extract_model` to extract subgraphs
- Uses ONNX Runtime to run the extracted model and generate expected outputs
- All paths are relative to the project root for consistency
- Docker ensures consistent Python/ONNX/ONNXRuntime versions regardless of host environment
