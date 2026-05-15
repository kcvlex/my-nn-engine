// Run under nsys for a per-kernel breakdown of the decode loop:
//
//   nsys profile \
//       --capture-range cudaProfilerApi \
//       --capture-range-end stop \
//       --output llama_decode \
//       --force-overwrite true \
//       cargo run --release -p my-nn-engine-llm --example profile_llama
//
// Then:
//   nsys stats --report cuda_kern_exec_sum llama_decode.nsqlite
//
// The cudaProfilerStart/Stop hooks bracket only the decode loop, so the ~10s
// session compile is excluded from the trace.

use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine_llm::build_llama;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::LlamaOptions;
use my_nn_engine_llm::LlamaWeights;
use my_nn_engine_llm::LlmSession;
use tokenizers::Tokenizer;

fn cuda_profiler_start() {
    unsafe {
        let lib = libloading::Library::new("libcudart.so").expect("failed to load libcudart.so");
        let func: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib
            .get(b"cudaProfilerStart")
            .expect("cudaProfilerStart not found");
        func();
        std::mem::forget(lib);
    }
}

fn cuda_profiler_stop() {
    unsafe {
        let lib = libloading::Library::new("libcudart.so").expect("failed to load libcudart.so");
        let func: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib
            .get(b"cudaProfilerStop")
            .expect("cudaProfilerStop not found");
        func();
        std::mem::forget(lib);
    }
}

fn main() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");

    let config = HfConfig::from_path(model_dir.join("config.json")).unwrap();
    let hf = HfWeights::from_dir(&model_dir).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 256;
    let n_warmup = 4;
    let n_measure = 32;

    let r = build_llama(&config, &weights, max_seq_len, &LlamaOptions::default());
    let opts = Options::builder().target(Target::CUDA).build();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut llm = LlmSession::for_llama(
        r.graph,
        r.kv_cache_names,
        tokenizer,
        &opts,
        max_seq_len,
        config.eos_token_id,
    )
    .unwrap();

    // Warmup so any one-time CUDA setup is excluded.
    let _ = llm.generate("The capital of France is", n_warmup).unwrap();
    llm.reset();

    cuda_profiler_start();
    let t0 = std::time::Instant::now();
    let _ = llm.generate("The capital of France is", n_measure).unwrap();
    let elapsed = t0.elapsed();
    cuda_profiler_stop();

    eprintln!(
        "{n_measure} new tokens in {:.2?}, avg {:.2?}/token",
        elapsed,
        elapsed / n_measure as u32
    );
}
