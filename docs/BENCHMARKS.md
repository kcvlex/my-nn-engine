# LLM Inference Benchmarks

Single-request decode/prefill on TinyLlama-1.1B and Llama2-7B-sft.
Median over 3 measured iterations after 1 warmup. Prompt
`"The capital of France is"`, `n_generate = 64`, greedy decode.
`max_seq_len = 256` for TinyLlama, `128` for Llama2-7B.

Measured at commit `4f27d8c` (branch `bench-llm`).

| runtime | model | dtype | decode tok/s | prefill tok/s | TTFT (ms) | peak VRAM (MiB) | peak RAM (MiB) |
|---|---|---|---:|---:|---:|---:|---:|
| my-nn-engine | TinyLlama-1.1B | BF16          | 70.0  | 3834.5 | 18.46 | 2569 | 721.1 |
| my-nn-engine | TinyLlama-1.1B | INT8 (W8A16)  | 68.3  | 128.1 | 139.56 | 1815 | 508.6 |
| llama.cpp    | TinyLlama-1.1B | Q8_0          | 127.7 | 1994.8 | 10.34 | 1503 | 1466.2 |
| ORT-GenAI    | TinyLlama-1.1B | FP16          | 68.8  | 5617.6 | 15.42 | 3529 | 49.9 |
| ORT-GenAI    | TinyLlama-1.1B | INT4          | 201.9 | 440.1 | 16.31 | 1485 | 50.1 |
| my-nn-engine | Llama2-7B-sft  | INT8 (W8A16)  | 15.0  | 21.2 | 820.86 | 7533 | 496.1 |
| llama.cpp    | Llama2-7B-sft  | Q8_0          | 22.2  | 297.0 | 61.93 | 7277 | 7180.2 |
| ORT-GenAI    | Llama2-7B-sft  | INT4          | 37.3  | 32.4 | 181.00 | 5633 | 49.9 |

Prefill columns measure different things between runtimes: my-nn-engine and
llama.cpp run a 16-token padded prefill chunk; ORT-GenAI consumes the 5-token
prompt directly with no padding. So per-token numbers within a row are
honest but cross-row prefill comparisons should be read with that in mind.

## Hardware

| component | spec |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation Laptop GPU (sm_89), 8188 MiB VRAM |
| GPU driver / CUDA | 595.71.05 / toolkit 13.2 |
| CPU | 13th Gen Intel Core i9-13900H (20 logical cores) |
| RAM | 62 GiB |
| Kernel | 7.0.2-arch1-1 |

How to reproduce: see [`crates/bench/README.md`](../crates/bench/README.md).
