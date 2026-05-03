"""ORT-GenAI baseline runner for the LLM bench.

Reads BENCH_* env vars (same convention as the Rust bench binaries) and emits
machine-parseable lines: ``BENCH_INFO`` (setup) and ``BENCH_ITER`` (per iteration).
Two timed passes per iteration are used so TTFT can be reported separately:

* pass A: generate(prompt, 1) -> prefill + 1 decode = TTFT proxy
* pass B: generate(prompt, n_generate)             = prefill + n decodes

The outer orchestrator polls VRAM/RSS independently.
"""
import os
import sys
import time

import onnxruntime_genai as og


def env_str(key, default):
    return os.environ.get(key, default)


def env_int(key, default):
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def main():
    model_dir = env_str("BENCH_MODEL_DIR", "")
    if not model_dir or not os.path.isdir(model_dir):
        print(f"BENCH_ERROR model_dir not found: {model_dir!r}", flush=True)
        sys.exit(2)

    label = env_str("BENCH_LABEL", os.path.basename(model_dir.rstrip("/")))
    n_generate = env_int("BENCH_N_GENERATE", 64)
    warmup = env_int("BENCH_WARMUP", 1)
    iters = env_int("BENCH_ITERS", 3)
    prompt = env_str("BENCH_PROMPT", "The capital of France is")
    max_seq_len = env_int("BENCH_MAX_SEQ_LEN", 256)

    print(
        f"BENCH_INFO pid={os.getpid()} runtime=ort-genai label={label} "
        f"n_generate={n_generate} warmup={warmup} iters={iters} max_seq_len={max_seq_len}",
        flush=True,
    )
    print(f"BENCH_INFO prompt={prompt!r}", flush=True)

    load_t0 = time.perf_counter()
    model = og.Model(model_dir)
    tok = og.Tokenizer(model)
    print(f"BENCH_INFO model_load_ms={(time.perf_counter() - load_t0) * 1e3:.3f}", flush=True)

    input_tokens = tok.encode(prompt)
    prompt_tokens = len(input_tokens)
    print(f"BENCH_INFO prompt_tokens={prompt_tokens}", flush=True)

    def run_generate(target_new_tokens):
        params = og.GeneratorParams(model)
        # Greedy / deterministic so we measure compute, not sampling.
        try:
            params.set_search_options(
                max_length=prompt_tokens + target_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        except TypeError:
            params.set_search_options(max_length=prompt_tokens + target_new_tokens)
        gen = og.Generator(model, params)
        gen.append_tokens(input_tokens)
        produced = 0
        while produced < target_new_tokens and not gen.is_done():
            gen.generate_next_token()
            produced += 1
        return produced, gen.get_sequence(0)

    sample_printed = False
    for i in range(warmup + iters):
        kind = "warmup" if i < warmup else "measure"
        idx = i if i < warmup else i - warmup

        t0 = time.perf_counter()
        ttft_n, _ = run_generate(1)
        ttft_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        full_n, full_seq = run_generate(n_generate)
        total_ms = (time.perf_counter() - t0) * 1e3

        if not sample_printed:
            try:
                tail = full_seq[prompt_tokens:]
                txt = tok.decode(tail)
                print(f"BENCH_INFO sample_output={txt!r}", flush=True)
            except Exception as e:
                print(f"BENCH_INFO sample_decode_error={e!r}", flush=True)
            sample_printed = True

        print(
            f"BENCH_ITER kind={kind} iter={idx} ttft_ms={ttft_ms:.3f} total_ms={total_ms:.3f} "
            f"n_generate={n_generate} gen_tokens={full_n}",
            flush=True,
        )


if __name__ == "__main__":
    main()
