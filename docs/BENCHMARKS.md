# LLM Inference Benchmarks

Single-request decode/prefill on TinyLlama-1.1B and Llama2-7B-hf.
Median over 3 measured iterations after 1 warmup. Prompt is a ~200-word
Lorem Ipsum passage (about 256 tokens for the Llama tokenizer),
`n_generate = 64`, greedy decode. `max_seq_len = 640` for all rows
(prefill 512 + generate 64 + margin).

Measured at commit `81221f1`.

| runtime | model | dtype | decode tok/s | prefill tok/s | TTFT (ms) | peak VRAM (MiB) | peak RAM (MiB) |
|-|-|-|-:|-:|-:|-:|-:|
| my-nn-engine | TinyLlama-1.1B | BF16 | 54.4 | 2836.8 | 198.87 | 2392 | 814.6 |
| my-nn-engine | TinyLlama-1.1B | INT8 (W8A16) | 50.8 | 1419.5 | 380.38 | 1634 | 620.4 |
| my-nn-engine | Llama2-7B-hf | INT8 (W8A16) | 14.3 | 255.3 | 2075.18 | 7682 | 623.2 |
| llama.cpp | TinyLlama-1.1B | Q8_0 | 128.6 | 5900.4 | 51.16 | 1360 | 1466.9 |
| llama.cpp | Llama2-7B-hf | Q8_0 | 22.4 | 1308.9 | 240.31 | 7282 | 7181.1 |
| ORT-GenAI | TinyLlama-1.1B | FP16 | 66.9 | 8881.4 | 43.77 | 3462 | 49.6 |
| ORT-GenAI | TinyLlama-1.1B | INT4 | 189.3 | 5241.9 | 54.12 | 1457 | 49.7 |
| ORT-GenAI | Llama2-7B-hf | INT4 | 36.3 | 716.8 | 384.69 | 5955 | 49.7 |

Prefill columns: my-nn-engine and llama.cpp run a 512-token padded prefill
chunk; ORT-GenAI consumes the prompt directly (about 256 tokens, no
padding). Per-token numbers within a row are honest but cross-row prefill
comparisons should be read with that in mind. The dtype column also varies
across runtimes (mynn INT8 vs llama.cpp Q8_0 vs ORT-GenAI INT4 / FP16),
so dtype-mismatched rows are reference points rather than apples-to-apples.

## Hardware

| component | spec |
|-|-|
| GPU | NVIDIA RTX 2000 Ada Generation Laptop GPU (sm_89), 8188 MiB VRAM |
| GPU driver / CUDA | 595.71.05 / toolkit 13.2 |
| CPU | 13th Gen Intel Core i9-13900H (20 logical cores) |
| RAM | 62 GiB |
| Kernel | 7.0.2-arch1-1 |

How to reproduce: see [`crates/bench-llm/README.md`](../crates/bench-llm/README.md).
