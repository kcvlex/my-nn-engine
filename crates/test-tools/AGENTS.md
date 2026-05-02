# my-nn-engine-test-tools — Agent Guide

This tool identifies which ONNX node first produces incorrect output by binary-searching over the model's node list. It extracts subgraphs inside a Podman container (with onnxruntime as reference), then runs the test command on the host.

## Prerequisites

- Podman installed and accessible
- The Podman image is auto-built on first run (no manual step needed)

## Build

Run both from the **project root** (`/home/kcvlex/my-nn-engine`):

```bash
cargo build -p my-nn-engine-test-tools --release
cargo build -p my-nn-engine-test-tools --example test_extracted --release
```

## Binary Search Usage

**All paths must be relative to the project root.** The tool mounts the project root as `/workspace` inside the container. Absolute host paths will not resolve inside the container.

**All commands must be run from the project root directory.**

```bash
cd /home/kcvlex/my-nn-engine

TARGET=CUDA EPSILON=0.01 ./target/release/my-nn-engine-test-tools \
  --mode binary-search \
  --model-path models/validated/GPT2/model.onnx \
  --input-paths models/validated/GPT2/test_data_set_0/input_0.pb \
  --test-command ./target/release/examples/test_extracted
```

### Environment Variables

These are inherited by the test command (not consumed by the binary search tool itself):

| Variable | Read by | Default | Description |
|---|---|---|---|
| `TARGET` | `test_extracted` | `CPU` | `CPU` or `CUDA` |
| `EPSILON` | `test_extracted` | `0.01` | Float comparison tolerance |
| `EXTRACTED_MODEL_DIR` | `test_extracted` | *(set by tool)* | Path to extracted subgraph directory (auto-set, do not override) |

### Model Paths

Available validated models (relative to project root):

| Model | model-path | input-paths |
|---|---|---|
| GPT-2 | `models/validated/GPT2/model.onnx` | `models/validated/GPT2/test_data_set_0/input_0.pb` |
| BERT | `models/validated/bertsquad-12/bertsquad-12.onnx` | `models/validated/bertsquad-12/test_data_set_0/input_0.pb models/validated/bertsquad-12/test_data_set_0/input_1.pb models/validated/bertsquad-12/test_data_set_0/input_2.pb models/validated/bertsquad-12/test_data_set_0/input_3.pb` |
| MNIST | `models/validated/mnist-12/mnist-12.onnx` | `models/validated/mnist-12/test_data_set_0/input_0.pb` |
| ResNet-18 | `models/validated/resnet18-v2-7/resnet18-v2-7.onnx` | `models/validated/resnet18-v2-7/test_data_set_0/input_0.pb` |
| ResNet-152 | `models/validated/resnet152-v2-7/resnet152-v2-7.onnx` | `models/validated/resnet152-v2-7/test_data_set_0/input_0.pb` |
| YOLOv4 | `models/validated/yolov4/yolov4.onnx` | `models/validated/yolov4/test_data_set_0/input_0.pb` |

To discover input files for a model: `ls models/validated/<MODEL>/test_data_set_0/input_*.pb`

### Output

The tool prints progress like:
```
Testing node [50/200] MatMul (node_name)
  PASS
Testing node [125/200] Softmax (node_name)
  FAIL
...
================================================================================
FOUND: First failing node
================================================================================
Index:    100
Node:     Add (some_add_node)
Extracted: /tmp/.tmpXXXXXX/node_100
```

The `Extracted` directory contains `model.onnx`, `input_*.pb`, and `output_*.pb` for the failing subgraph. Use these for further debugging.

### Limitations

The binary search tests each node as the sole output of a subgraph. If a node only outputs shape/constant data (e.g., Shape, Gather, Unsqueeze, Concat producing shape tensors), it will always PASS even if upstream data-path nodes are wrong. The reported "first failing node" is the first node whose output is a full data tensor that exposes the error. The actual buggy node may be earlier.

## Manual Node Extraction (via Podman directly)

To test a specific node index without running the full binary search, call the container's `binary_search.py extract-node` directly:

```bash
# output-dir MUST be relative to project root (inside mounted volume)
podman run --rm --userns=keep-id \
  --entrypoint python \
  -v /home/kcvlex/my-nn-engine:/workspace -w /workspace \
  my-nn-engine-test-tools /usr/local/bin/binary_search.py \
  extract-node \
  --model models/validated/GPT2/model.onnx \
  --inputs models/validated/GPT2/test_data_set_0/input_0.pb \
  --node-index 1894 \
  --output-dir tmp/gpt2_debug/node_1894
```

Then test on host:

```bash
EXTRACTED_MODEL_DIR=/home/kcvlex/my-nn-engine/tmp/gpt2_debug/node_1894 \
  TARGET=CUDA EPSILON=0.01 \
  /home/kcvlex/my-nn-engine/target/release/examples/test_extracted
```

## Extraction Mode (by output node name)

Extract a subgraph up to a named output node:

```bash
./target/release/my-nn-engine-test-tools \
  --mode extract \
  --model-path models/validated/GPT2/model.onnx \
  --input-paths models/validated/GPT2/test_data_set_0/input_0.pb \
  --output-node-name "output_node_name" \
  --output-dir models/extracted/GPT2/subgraph_name
```

## Architecture

- **Container** (Podman): Runs Python with onnx/onnxruntime. Handles model extraction (`onnx.utils.extract_model`) and reference inference. Mounted volume: project root -> `/workspace`.
- **Host**: Runs the test command binary (e.g. `test_extracted`), which loads the extracted model with my-nn-engine's `Session` and compares outputs.
- The binary search tool orchestrates both: it calls the container for extraction, then the host binary for testing.
