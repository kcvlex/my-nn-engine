use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine_llm::build_llama;
use my_nn_engine_llm::quantize::quantize_safetensors_int8_dir;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::LlamaWeights;
use my_nn_engine_llm::LlmSession;
use tokenizers::Tokenizer;

fn main() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/llama2-7b-sft");

    let config = HfConfig::from_path(model_dir.join("config.json")).unwrap();
    let safetensors_arg = std::env::args().nth(1);
    let hf = match safetensors_arg.as_deref() {
        Some(name) => HfWeights::from_safetensors(model_dir.join(name)).unwrap(),
        None => {
            let int8_path = model_dir.join("model.int8.safetensors");
            if !int8_path.exists() {
                println!("quantizing to INT8 (one-time, ~30s)...");
                let t0 = std::time::Instant::now();
                let stats = quantize_safetensors_int8_dir(&model_dir, &int8_path).unwrap();
                println!(
                    "  quantized {} tensors, passed through {} ({:.2?})",
                    stats.quantized,
                    stats.passthrough,
                    t0.elapsed()
                );
            }
            HfWeights::from_safetensors(&int8_path).unwrap()
        }
    };
    println!("loaded {} tensors", hf.names().count());
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 128;
    let n_generate = 24;

    println!("building decode graph...");
    let t0 = std::time::Instant::now();
    let r = build_llama(&config, &weights, max_seq_len);
    println!("  decode graph built in {:.2?}", t0.elapsed());

    println!("compiling session (CUDA)...");
    let t0 = std::time::Instant::now();
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
    println!("  compiled in {:.2?}", t0.elapsed());

    let prompt = "The capital of France is";
    let t0 = std::time::Instant::now();
    let text = llm.generate(prompt, n_generate).unwrap();
    println!("decode in {:.2?}", t0.elapsed());
    println!("prompt: {prompt:?}");
    println!("output: {text}");
}
