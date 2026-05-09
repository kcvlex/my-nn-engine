# my-nn-bench-llm

Orchestrator that runs side-by-side LLM benchmarks against my-nn-engine,
llama.cpp, and ORT-GenAI; aggregates timing, peak VRAM, and peak RSS into
a markdown table.

## Usage

```sh
# Run all configured rows (~5-10 minutes on the reference machine).
cargo run --release -p my-nn-bench-llm

# Quick mode (1 measured iter instead of 3).
cargo run --release -p my-nn-bench-llm -- --quick

# Filter rows by substring of label.
cargo run --release -p my-nn-bench-llm -- --filter llama2
```

Output: `target/bench/results/SUMMARY.md` and per-row raw logs under
`target/bench/raw/`.

Environment overrides:

- `LLAMACPP_BIN` (default `~/baselines/llama.cpp/build/bin`)
- `GGUF_DIR` (default `~/baselines/gguf`)
- `ORTGENAI_DIR` (default `~/baselines/ortgenai`)
- `ORTGENAI_IMAGE` (default `ortgenai-bench`)

## Setup

### Container

The ORT-GenAI baseline and the llama.cpp `convert_hf_to_gguf.py` script
both run inside a CUDA container built from this crate's Dockerfile:

```sh
podman build -t ortgenai-bench crates/bench-llm/
```

Bundled tooling: `onnxruntime-genai-cuda==0.11.4`, `transformers`, `torch`,
`onnx`, `onnx-ir`, `sentencepiece`.

### Models (mynn)

mynn reads HF safetensors layouts under `models/hf/`. Place each model
under `models/hf/<name>/` (config.json + tokenizer + safetensors). The
INT8 file is auto-quantized on first run if missing; pre-quantize via
`my-nn-engine-llm`'s `quantize_int8` binary.

### llama.cpp baseline

Build llama.cpp once with CUDA:

```sh
git clone https://github.com/ggerganov/llama.cpp ~/baselines/llama.cpp
cmake -B ~/baselines/llama.cpp/build \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF \
      -S ~/baselines/llama.cpp
cmake --build ~/baselines/llama.cpp/build --config Release -j 16 \
      --target llama-cli llama-bench llama-quantize
```

Convert HF -> GGUF (F16) -> Q8_0 via the container:

```sh
mkdir -p ~/baselines/gguf
podman run --rm \
  -v <hf-model-dir>:/model:ro \
  -v $HOME/baselines/gguf:/output \
  -v $HOME/baselines/llama.cpp:/llama_cpp:ro \
  ortgenai-bench \
  python3 /llama_cpp/convert_hf_to_gguf.py /model \
    --outfile /output/<name>-f16.gguf --outtype f16

~/baselines/llama.cpp/build/bin/llama-quantize \
  ~/baselines/gguf/<name>-f16.gguf \
  ~/baselines/gguf/<name>-q8_0.gguf q8_0
```

### ORT-GenAI baseline

Build per-model dirs with `onnxruntime_genai.models.builder` inside the
container:

```sh
mkdir -p ~/baselines/ortgenai
podman run --rm --device nvidia.com/gpu=all \
  -v <hf-model-dir>:/model:ro \
  -v $HOME/baselines/ortgenai:/output \
  -v /tmp/ortgenai-cache:/cache \
  ortgenai-bench \
  python3 -m onnxruntime_genai.models.builder \
    -i /model -o /output/<name>-fp16 -p fp16 -e cuda -c /cache
```

`my-nn-bench-llm` silently skips any row whose ortgenai dir is missing.

## Results

Per-row raw logs are written under `target/bench/raw/`, and the aggregated markdown table at `target/bench/results/SUMMARY.md`.
