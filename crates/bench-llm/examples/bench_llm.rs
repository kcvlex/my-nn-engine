// Unified bench binary for my-nn-engine LLM examples.
//
// Reads config from env vars:
//   BENCH_MODEL_DIR    = path to HF model dir (required)
//   BENCH_LABEL        = string (used in BENCH_INFO; default basename of MODEL_DIR)
//   BENCH_DTYPE        = "bf16" | "int8"  (default bf16)
//   BENCH_N_GENERATE   = usize  (default 64)
//   BENCH_WARMUP       = usize  (default 1)
//   BENCH_ITERS        = usize  (default 3)
//   BENCH_PROMPT       = string (default "The capital of France is")
//   BENCH_MAX_SEQ_LEN  = usize  (default 256)
//   BENCH_PREFILL_LEN  = usize  (default 16; set to 0 for decode-only)
//   BENCH_AUTO_QUANT   = "1" to auto-generate model.int8.safetensors when missing
//
// Emits machine-parseable lines (`BENCH_INFO`, `BENCH_ITER`) for the orchestrator.
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use my_nn_engine::options::Options;
use my_nn_engine::options::PrefetchPolicy;
use my_nn_engine::options::Target;
use my_nn_engine_llm::build_decoder;
use my_nn_engine_llm::quantize::quantize_safetensors_int8_dir;
use my_nn_engine_llm::BuildOptions;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::LlmSession;
use my_nn_engine_llm::ModelSpec;
use tokenizers::Tokenizer;

fn env_str(k: &str, default: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| default.to_string())
}
fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn env_bool(k: &str) -> bool {
    matches!(
        std::env::var(k).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

fn resolve_safetensors(model_dir: &Path, dtype: &str, auto_quant: bool) -> PathBuf {
    match dtype {
        "bf16" => model_dir.join("model.safetensors"),
        "int8" => {
            let p = model_dir.join("model.int8.safetensors");
            if !p.exists() && auto_quant {
                let stats = quantize_safetensors_int8_dir(model_dir, &p).unwrap();
                println!(
                    "BENCH_INFO quantized={} passthrough={}",
                    stats.quantized, stats.passthrough,
                );
            }
            p
        }
        other => panic!("BENCH_DTYPE must be bf16 or int8 (got {other:?})"),
    }
}

fn main() {
    let model_dir = PathBuf::from(env_str(
        "BENCH_MODEL_DIR",
        "BENCH_MODEL_DIR is required (e.g. models/hf/tinyllama)",
    ));
    let label = env_str(
        "BENCH_LABEL",
        model_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model"),
    );
    let dtype = env_str("BENCH_DTYPE", "bf16");
    let n_generate = env_usize("BENCH_N_GENERATE", 64);
    let warmup = env_usize("BENCH_WARMUP", 1);
    let iters = env_usize("BENCH_ITERS", 3);
    let prompt = env_str("BENCH_PROMPT", "The capital of France is");
    let max_seq_len = env_usize("BENCH_MAX_SEQ_LEN", 256);
    let prefill_len = env_usize("BENCH_PREFILL_LEN", 16);
    let auto_quant = env_bool("BENCH_AUTO_QUANT");

    println!(
        "BENCH_INFO pid={} label={} dtype={} n_generate={} warmup={} iters={} \
         max_seq_len={} prefill_len={}",
        std::process::id(),
        label,
        dtype,
        n_generate,
        warmup,
        iters,
        max_seq_len,
        prefill_len,
    );
    println!("BENCH_INFO prompt={prompt:?}");

    let load_t0 = Instant::now();
    let config = HfConfig::from_path(model_dir.join("config.json")).unwrap();
    let st_path = resolve_safetensors(&model_dir, &dtype, auto_quant);
    let hf = HfWeights::from_safetensors(&st_path).unwrap();
    let spec = ModelSpec::from_hf(&config, &hf).unwrap();
    println!(
        "BENCH_INFO weight_load_ms={:.3}",
        load_t0.elapsed().as_secs_f64() * 1e3
    );

    let r = build_decoder(&config, &spec, max_seq_len, &BuildOptions::default());
    let quantize_activations = std::env::var("BENCH_QUANTIZE_ACTIVATIONS").is_ok();
    let num_cuda_streams: usize = std::env::var("BENCH_NUM_CUDA_STREAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let opts = Options::builder()
        .target(Target::CUDA(PrefetchPolicy::Disabled))
        .quantize_activations(quantize_activations)
        .num_cuda_streams(num_cuda_streams)
        .build();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let compile_t0 = Instant::now();
    let mut llm = if prefill_len > 0 {
        let p = build_decoder(
            &config,
            &spec,
            max_seq_len,
            &BuildOptions::builder().prefill_len(prefill_len).build(),
        );
        LlmSession::for_llama_with_prefill(
            r.graph,
            p.graph,
            r.kv_cache_names,
            prefill_len,
            tokenizer,
            &opts,
            max_seq_len,
            config.eos_token_id,
        )
        .unwrap()
    } else {
        LlmSession::for_llama(
            r.graph,
            r.kv_cache_names,
            tokenizer,
            &opts,
            max_seq_len,
            config.eos_token_id,
        )
        .unwrap()
    };
    println!(
        "BENCH_INFO compile_ms={:.3}",
        compile_t0.elapsed().as_secs_f64() * 1e3
    );

    let aux_tok = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let prompt_tokens = aux_tok
        .encode(prompt.as_str(), false)
        .unwrap()
        .get_ids()
        .len();
    let prefill_padded = if prefill_len > 0 {
        prompt_tokens.div_ceil(prefill_len) * prefill_len
    } else {
        0
    };
    println!("BENCH_INFO prompt_tokens={prompt_tokens} prefill_padded_tokens={prefill_padded}");

    let mut printed_sample = false;
    for i in 0..(warmup + iters) {
        let kind = if i < warmup { "warmup" } else { "measure" };
        let iter_idx = if i < warmup { i } else { i - warmup };

        // Pass 1: TTFT (prefill + 1 decode step).
        llm.reset();
        let t0 = Instant::now();
        let _ = llm.generate_ids(&prompt, 1).unwrap();
        let ttft_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Pass 2: full generation.
        llm.reset();
        let t0 = Instant::now();
        let ids = llm.generate_ids(&prompt, n_generate).unwrap();
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;
        let gen_tokens = ids.len();

        if !printed_sample {
            if let Ok(text) = aux_tok.decode(&ids, false) {
                println!("BENCH_INFO sample_output={text:?}");
            }
            printed_sample = true;
        }

        println!(
            "BENCH_ITER kind={kind} iter={iter_idx} ttft_ms={ttft_ms:.3} total_ms={total_ms:.3} \
             n_generate={n_generate} gen_tokens={gen_tokens}",
        );
    }
}
