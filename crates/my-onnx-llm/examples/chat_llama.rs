// Multi-turn chat using TinyLlama's chat template + KV-cache continuation.
// Each turn renders <|user|> / <|assistant|> sections and feeds only the new
// suffix to the model so the cache from the previous turn is reused.

use std::path::PathBuf;

use my_onnx::options::Options;
use my_onnx::options::Target;
use my_onnx_llm::apply_chat_template;
use my_onnx_llm::build_llama;
use my_onnx_llm::llama::build_llama_prefill;
use my_onnx_llm::ChatMessage;
use my_onnx_llm::HfConfig;
use my_onnx_llm::HfWeights;
use my_onnx_llm::LlamaWeights;
use my_onnx_llm::LlmSession;
use tokenizers::Tokenizer;

fn main() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");
    let template = std::fs::read_to_string(model_dir.join("chat_template.jinja")).unwrap();

    let config = HfConfig::from_path(model_dir.join("config.json")).unwrap();
    let hf = HfWeights::from_index(
        model_dir.join("weights.f32.bin"),
        model_dir.join("weights.f32.json"),
    )
    .unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 512;
    let prefill_len = 64;

    let r = build_llama(&config, &weights, max_seq_len);
    let p = build_llama_prefill(&config, &weights, max_seq_len, prefill_len);
    let opts = Options::builder().target(Target::CUDA).build();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut llm = LlmSession::for_llama_with_prefill(
        r.graph,
        p.graph,
        r.kv_cache_names,
        prefill_len,
        tokenizer,
        &opts,
        max_seq_len,
        config.eos_token_id,
    )
    .unwrap();

    let mut history: Vec<ChatMessage> = vec![ChatMessage::system(
        "You are a friendly, helpful assistant. Keep replies short.",
    )];
    let user_turns = ["Hello, my name is Alice.", "What name did I just tell you?"];

    let mut last_rendered_len = 0;
    for turn in user_turns {
        history.push(ChatMessage::user(turn));
        let rendered = apply_chat_template(&template, &history, "</s>", "<s>", true).unwrap();
        let new_suffix = &rendered[last_rendered_len..];
        let reply = llm.generate(new_suffix, 64).unwrap();
        println!("user: {turn}");
        println!("asst: {reply}\n");
        history.push(ChatMessage::assistant(reply.trim().to_string()));
        last_rendered_len = apply_chat_template(&template, &history, "</s>", "<s>", false)
            .unwrap()
            .len();
    }
}
