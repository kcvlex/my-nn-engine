# my-nn-engine-llm

LLM runtime built on top of [`my-nn-engine`](../..).

## Features

- Model builders for the Llama family of LLMs against the `my-nn-engine` graph IR.
- INT8 weight quantization with sharded streaming.
- Chunked prefill + KV cache fusion + INT8 KV cache.

## Quickstart

```sh
# Download a model from HuggingFace Hub (gated; HF_TOKEN may be needed).
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir model-dir

# INT8 quantize. Positional args are <input_dir> <output_dir>
cargo run --release -p my-nn-engine-llm --bin quantize_int8 -- model-dir model-dir

# Run an example end-to-end on TinyLlama.
cargo run --release -p my-nn-engine-llm --example run_tinyllama
```

## Benchmarks

See [`BENCHMARKS.md`](../../docs/BENCHMARKS.md) for details.

## Roadmap

- INT4 quantization (AWQ / GPTQ / custom; design open).
- CPU/GPU hybrid offload (FFN on CPU, attention on GPU).
