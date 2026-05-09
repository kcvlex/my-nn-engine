// Interactive multi-turn chat using a Llama chat template + KV-cache continuation.
// Each turn renders <|user|> / <|assistant|> sections and feeds only the new
// suffix to the model so the cache from the previous turn is reused.
//
// Usage:
//   cargo run --release -p my-nn-engine-llm --example chat_llama -- [<model_dir>] [cpu|cuda]
//
// Type a line and press Enter to send. Ctrl+D (EOF), an empty line, or
// `exit` / `quit` ends the session.

use std::io::BufRead;
use std::io::Write;
use std::io::{self};
use std::path::Path;
use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine_llm::apply_chat_template;
use my_nn_engine_llm::build_llama;
use my_nn_engine_llm::llama::build_llama_prefill;
use my_nn_engine_llm::ChatMessage;
use my_nn_engine_llm::GenerateOptions;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::LlamaWeights;
use my_nn_engine_llm::LlmSession;
use serde::Deserialize;
use tokenizers::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama")
    });
    let target = match args.next().as_deref() {
        Some("cpu") => Target::CPU,
        Some("cuda") | None => Target::CUDA,
        Some(other) => panic!("unknown target {other:?} (expected `cpu` or `cuda`)"),
    };

    let template = std::fs::read_to_string(model_dir.join("chat_template.jinja"))
        .expect("chat_template.jinja required in model dir");
    let config = HfConfig::from_path(model_dir.join("config.json")).unwrap();
    let hf = HfWeights::from_dir(&model_dir).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 512;
    let prefill_len = 64;

    let r = build_llama(&config, &weights, max_seq_len);
    let p = build_llama_prefill(&config, &weights, max_seq_len, prefill_len);
    let opts = Options::builder().target(target).build();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let eos_str = tokenizer
        .decode(&[config.eos_token_id], false)
        .unwrap_or_else(|_| "</s>".to_string());
    let stop_strings = load_stop_strings(&model_dir, &eos_str, &tokenizer);
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
    let mut last_rendered_len = 0;

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("user> ");
        stdout.flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            println!();
            break;
        }
        let user_input = line.trim();
        if user_input.is_empty() || user_input == "exit" || user_input == "quit" {
            break;
        }

        history.push(ChatMessage::user(user_input));
        let rendered = apply_chat_template(&template, &history, "</s>", "<s>", true).unwrap();
        let new_suffix = &rendered[last_rendered_len..];
        let opts = GenerateOptions {
            max_new_tokens: 256,
            stop_strings: stop_strings.clone(),
        };
        let reply = llm.generate_with(new_suffix, &opts).unwrap();
        println!("asst> {}\n", reply.trim());
        history.push(ChatMessage::assistant(reply.trim().to_string()));
        last_rendered_len = apply_chat_template(&template, &history, "</s>", "<s>", false)
            .unwrap()
            .len();
    }
}

#[derive(Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: Option<EosIds>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EosIds {
    One(u32),
    Many(Vec<u32>),
}

fn load_stop_strings(model_dir: &Path, eos_token_str: &str, tokenizer: &Tokenizer) -> Vec<String> {
    let mut out: Vec<String> = vec![eos_token_str.to_string()];
    let gen_path = model_dir.join("generation_config.json");
    if let Ok(s) = std::fs::read_to_string(&gen_path) {
        if let Ok(parsed) = serde_json::from_str::<GenerationConfig>(&s) {
            let ids: Vec<u32> = match parsed.eos_token_id {
                Some(EosIds::One(x)) => vec![x],
                Some(EosIds::Many(xs)) => xs,
                None => vec![],
            };
            for id in ids {
                if let Some(tok) = tokenizer.id_to_token(id) {
                    out.push(tok);
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}
