// LLM benchmark orchestrator - runs my-nn-engine + llama.cpp + ORT-GenAI
// baselines, polls VRAM/RSS, aggregates timings into a markdown table.
//
// CLI flags:
//   --quick    run 1 measured iter (debug); default is 3
//   --filter <substr>   only run rows whose label contains <substr>
//
// Env overrides:
//   LLAMACPP_BIN   default ~/baselines/llama.cpp/build/bin
//   GGUF_DIR       default ~/baselines/gguf
//   ORTGENAI_DIR   default ~/baselines/ortgenai
//   ORTGENAI_IMAGE default ortgenai-bench

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const PROMPT: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident.";
const PROMPT_TOKENS: u32 = 256;
const N_GENERATE: u32 = 64;
const POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone)]
struct RunSpec {
    label: String,
    runtime: Runtime,
}

#[derive(Debug, Clone)]
enum Runtime {
    Mynn {
        model_dir: PathBuf,
        dtype: &'static str,
        max_seq_len: u32,
        prefill_len: u32,
        prefill_padded_for_aggr: u32,
    },
    LlamaCpp {
        gguf: PathBuf,
        prefill_n: u32, // 0 = decode-only
        gen_n: u32,
    },
    OrtGenAI {
        model_dir: PathBuf,
        max_seq_len: u32,
    },
}

#[derive(Debug, Default)]
struct Aggregate {
    median_ttft_ms: f64,
    decode_tok_s: f64,
    prefill_tok_s: Option<f64>,
    peak_vram_mib: u64,
    peak_rss_mib: f64,
    sample_output: Option<String>,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}
fn env_path(key: &str, default: PathBuf) -> PathBuf {
    std::env::var(key).map(PathBuf::from).unwrap_or(default)
}

fn build_runs(repo_root: &Path) -> Vec<RunSpec> {
    let llamacpp_bin = env_path("LLAMACPP_BIN", home().join("baselines/llama.cpp/build/bin"));
    let gguf_dir = env_path("GGUF_DIR", home().join("baselines/gguf"));
    let ortgenai_dir = env_path("ORTGENAI_DIR", home().join("baselines/ortgenai"));
    let _ = llamacpp_bin; // expanded in build_command

    let models = repo_root.join("models/hf");
    let mut runs = vec![
        RunSpec {
            label: "mynn-tinyllama-bf16".into(),
            runtime: Runtime::Mynn {
                model_dir: models.join("tinyllama"),
                dtype: "bf16",
                max_seq_len: 640,
                prefill_len: 512,
                prefill_padded_for_aggr: 512,
            },
        },
        RunSpec {
            label: "mynn-tinyllama-int8".into(),
            runtime: Runtime::Mynn {
                model_dir: models.join("tinyllama"),
                dtype: "int8",
                max_seq_len: 640,
                prefill_len: 512,
                prefill_padded_for_aggr: 512,
            },
        },
        RunSpec {
            label: "mynn-llama2-int8".into(),
            runtime: Runtime::Mynn {
                model_dir: models.join("llama2-7b-hf"),
                dtype: "int8",
                max_seq_len: 640,
                prefill_len: 512,
                prefill_padded_for_aggr: 512,
            },
        },
        RunSpec {
            label: "llamacpp-tinyllama-q8_0".into(),
            runtime: Runtime::LlamaCpp {
                gguf: gguf_dir.join("tinyllama-q8_0.gguf"),
                prefill_n: 512,
                gen_n: 64,
            },
        },
        RunSpec {
            label: "llamacpp-llama2-q8_0".into(),
            runtime: Runtime::LlamaCpp {
                gguf: gguf_dir.join("llama2-7b-hf-q8_0.gguf"),
                prefill_n: 512,
                gen_n: 64,
            },
        },
    ];

    for (sub, label, max_seq) in [
        ("tinyllama-fp16", "ortgenai-tinyllama-fp16", 640u32),
        ("tinyllama-int4", "ortgenai-tinyllama-int4", 640),
        ("llama2-7b-int4", "ortgenai-llama2-7b-int4", 640),
    ] {
        let dir = ortgenai_dir.join(sub);
        if dir.is_dir() {
            runs.push(RunSpec {
                label: label.into(),
                runtime: Runtime::OrtGenAI {
                    model_dir: dir,
                    max_seq_len: max_seq,
                },
            });
        }
    }

    runs
}

fn build_command(spec: &RunSpec, repo_root: &Path, warmup: u32, iters: u32) -> Command {
    match &spec.runtime {
        Runtime::Mynn {
            model_dir,
            dtype,
            max_seq_len,
            prefill_len,
            ..
        } => {
            let bin = repo_root.join("target/release/examples/bench_llm");
            let mut c = Command::new(bin);
            c.env("BENCH_MODEL_DIR", model_dir)
                .env("BENCH_LABEL", &spec.label)
                .env("BENCH_DTYPE", *dtype)
                .env("BENCH_N_GENERATE", N_GENERATE.to_string())
                .env("BENCH_WARMUP", warmup.to_string())
                .env("BENCH_ITERS", iters.to_string())
                .env("BENCH_PROMPT", PROMPT)
                .env("BENCH_MAX_SEQ_LEN", max_seq_len.to_string())
                .env("BENCH_PREFILL_LEN", prefill_len.to_string())
                .env("BENCH_AUTO_QUANT", "1");
            c
        }
        Runtime::LlamaCpp {
            gguf,
            prefill_n,
            gen_n,
        } => {
            let llamacpp_bin =
                env_path("LLAMACPP_BIN", home().join("baselines/llama.cpp/build/bin"));
            let mut c = Command::new(llamacpp_bin.join("llama-bench"));
            c.arg("-m")
                .arg(gguf)
                .args(["-ngl", "99", "-r"])
                .arg(iters.to_string())
                .args(["-o", "json"])
                .args(["-p", &prefill_n.to_string()])
                .args(["-n", &gen_n.to_string()]);
            c
        }
        Runtime::OrtGenAI {
            model_dir,
            max_seq_len,
        } => {
            let image = std::env::var("ORTGENAI_IMAGE").unwrap_or_else(|_| "ortgenai-bench".into());
            let bench_py = repo_root.join("scripts/bench_ortgenai.py");
            let mut c = Command::new("podman");
            c.args(["run", "--rm", "--device", "nvidia.com/gpu=all"])
                .arg("-v")
                .arg(format!("{}:/model:ro", model_dir.display()))
                .arg("-v")
                .arg(format!(
                    "{}:/workspace/bench_ortgenai.py:ro",
                    bench_py.display()
                ))
                .args([
                    "-e",
                    "BENCH_MODEL_DIR=/model",
                    "-e",
                    &format!("BENCH_LABEL={}", spec.label),
                    "-e",
                    &format!("BENCH_N_GENERATE={N_GENERATE}"),
                    "-e",
                    &format!("BENCH_WARMUP={warmup}"),
                    "-e",
                    &format!("BENCH_ITERS={iters}"),
                    "-e",
                    &format!("BENCH_PROMPT={PROMPT}"),
                    "-e",
                    &format!("BENCH_MAX_SEQ_LEN={max_seq_len}"),
                ])
                .arg(image)
                .args(["python3", "/workspace/bench_ortgenai.py"]);
            c
        }
    }
}

fn poll_vram(stop: Arc<AtomicBool>) -> thread::JoinHandle<u64> {
    thread::spawn(move || {
        let mut peak: u64 = 0;
        while !stop.load(Ordering::Relaxed) {
            if let Ok(out) = Command::new("nvidia-smi")
                .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                .output()
            {
                if let Ok(s) = std::str::from_utf8(&out.stdout) {
                    if let Ok(v) = s.trim().parse::<u64>() {
                        peak = peak.max(v);
                    }
                }
            }
            thread::sleep(POLL_INTERVAL);
        }
        peak
    })
}

fn poll_rss(pid: u32, stop: Arc<AtomicBool>) -> thread::JoinHandle<u64> {
    thread::spawn(move || {
        let path = format!("/proc/{pid}/status");
        let mut peak_kb: u64 = 0;
        while !stop.load(Ordering::Relaxed) {
            if let Ok(s) = fs::read_to_string(&path) {
                for line in s.lines() {
                    if let Some(rest) = line.strip_prefix("VmRSS:") {
                        if let Some(num) = rest.split_whitespace().next() {
                            if let Ok(v) = num.parse::<u64>() {
                                peak_kb = peak_kb.max(v);
                            }
                        }
                    }
                }
            } else {
                break;
            }
            thread::sleep(POLL_INTERVAL);
        }
        peak_kb
    })
}

fn run_with_monitors(mut cmd: Command, log_path: &Path) -> std::io::Result<()> {
    let log_file = fs::File::create(log_path)?;
    let log_dup = log_file.try_clone()?;

    let stop = Arc::new(AtomicBool::new(false));
    let vram_handle = poll_vram(stop.clone());

    cmd.stdout(Stdio::from(log_file));
    cmd.stderr(Stdio::from(log_dup));
    let mut child = cmd.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            let prog = cmd.get_program().to_string_lossy().into_owned();
            let hint = if prog == "podman" {
                " (install podman from https://podman.io and re-run)"
            } else {
                ""
            };
            std::io::Error::new(e.kind(), format!("`{prog}` not found in PATH{hint}"))
        } else {
            e
        }
    })?;
    let pid = child.id();
    let rss_handle = poll_rss(pid, stop.clone());

    let _ = child.wait()?;
    stop.store(true, Ordering::Relaxed);

    let vram = vram_handle.join().unwrap_or(0);
    let rss = rss_handle.join().unwrap_or(0);

    // Side-channel files mirror the historical .vram.txt / .rss.txt outputs but
    // boiled down to peaks (the per-sample traces aren't useful downstream).
    let stem = log_path.with_extension("");
    fs::write(stem.with_extension("vram.txt"), format!("{vram}\n"))?;
    fs::write(stem.with_extension("rss.txt"), format!("{rss}\n"))?;
    Ok(())
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn aggregate_mynn(log: &str, prefill_pad: u32) -> Aggregate {
    let mut ttfts = Vec::new();
    let mut totals = Vec::new();
    let mut sample = None;
    let mut n_generate = 0f64;
    for line in log.lines() {
        if let Some(rest) = line.strip_prefix("BENCH_INFO sample_output=") {
            sample = Some(rest.trim_matches('"').to_string());
        }
        if line.contains("BENCH_ITER kind=measure") {
            for tok in line.split_whitespace() {
                if let Some(v) = tok.strip_prefix("ttft_ms=") {
                    if let Ok(x) = v.parse() {
                        ttfts.push(x);
                    }
                } else if let Some(v) = tok.strip_prefix("total_ms=") {
                    if let Ok(x) = v.parse() {
                        totals.push(x);
                    }
                } else if let Some(v) = tok.strip_prefix("n_generate=") {
                    if let Ok(x) = v.parse() {
                        n_generate = x;
                    }
                }
            }
        }
    }
    let mt = median(&mut ttfts);
    let mtot = median(&mut totals);
    let mut a = Aggregate::default();
    a.median_ttft_ms = mt;
    if n_generate > 1.0 && (mtot - mt) > 0.0 {
        let decode_step = (mtot - mt) / (n_generate - 1.0);
        a.decode_tok_s = 1000.0 / decode_step;
        let prefill_time = mt - decode_step;
        if prefill_pad > 0 && prefill_time > 0.0 {
            a.prefill_tok_s = Some(prefill_pad as f64 * 1000.0 / prefill_time);
        }
    }
    a.sample_output = sample;
    a
}

fn aggregate_llamacpp(log: &str, prompt_tokens: u32) -> Aggregate {
    let start = log.find('[');
    let end = log.rfind(']');
    let mut a = Aggregate::default();
    let (s, e) = match (start, end) {
        (Some(s), Some(e)) if e > s => (s, e),
        _ => return a,
    };
    let json_text = &log[s..=e];
    let parsed: serde_json::Value = match serde_json::from_str(json_text) {
        Ok(v) => v,
        Err(_) => return a,
    };
    let mut pp_ts: Vec<f64> = Vec::new();
    let mut tg_ts: Vec<f64> = Vec::new();
    if let Some(arr) = parsed.as_array() {
        for entry in arr {
            let n_prompt = entry.get("n_prompt").and_then(|v| v.as_u64()).unwrap_or(0);
            let n_gen = entry.get("n_gen").and_then(|v| v.as_u64()).unwrap_or(0);
            if let Some(samples) = entry.get("samples_ts").and_then(|v| v.as_array()) {
                for s in samples {
                    if let Some(f) = s.as_f64() {
                        if n_prompt > 0 {
                            pp_ts.push(f);
                        } else if n_gen > 0 {
                            tg_ts.push(f);
                        }
                    }
                }
            }
        }
    }
    let pp = median(&mut pp_ts);
    let tg = median(&mut tg_ts);
    if tg > 0.0 {
        a.decode_tok_s = tg;
        if pp > 0.0 {
            a.median_ttft_ms = (prompt_tokens as f64 / pp + 1.0 / tg) * 1000.0;
            a.prefill_tok_s = Some(pp);
        } else {
            // Decode-only: prompt is consumed token-by-token through the decode path.
            a.median_ttft_ms = prompt_tokens as f64 / tg * 1000.0;
        }
    }
    a
}

fn fmt_row(runtime: &str, model: &str, dtype: &str, a: &Aggregate) -> String {
    let pp = match a.prefill_tok_s {
        Some(v) if v > 0.0 => format!("{v:.1}"),
        _ => "-".into(),
    };
    format!(
        "| {runtime} | {model} | {dtype} | {decode:.1} | {pp} | {ttft:.2} | {vram} | {rss:.1} |",
        decode = a.decode_tok_s,
        ttft = a.median_ttft_ms,
        vram = a.peak_vram_mib,
        rss = a.peak_rss_mib,
    )
}

fn label_to_runtime_model_dtype(label: &str) -> (&'static str, &'static str, &'static str) {
    match label {
        "mynn-tinyllama-bf16" => ("my-nn-engine", "TinyLlama-1.1B", "BF16"),
        "mynn-tinyllama-int8" => ("my-nn-engine", "TinyLlama-1.1B", "INT8 (W8A16)"),
        "mynn-llama2-int8" => ("my-nn-engine", "Llama2-7B-hf", "INT8 (W8A16)"),
        "llamacpp-tinyllama-q8_0" => ("llama.cpp", "TinyLlama-1.1B", "Q8_0"),
        "llamacpp-llama2-q8_0" => ("llama.cpp", "Llama2-7B-hf", "Q8_0"),
        "ortgenai-tinyllama-fp16" => ("ORT-GenAI", "TinyLlama-1.1B", "FP16"),
        "ortgenai-tinyllama-int4" => ("ORT-GenAI", "TinyLlama-1.1B", "INT4"),
        "ortgenai-llama2-7b-int4" => ("ORT-GenAI", "Llama2-7B-hf", "INT4"),
        _ => ("?", "?", "?"),
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let filter = args
        .windows(2)
        .find(|w| w[0] == "--filter")
        .map(|w| w[1].clone());
    let warmup: u32 = 1;
    let iters: u32 = if quick { 1 } else { 3 };

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let raw_dir = repo_root.join("target/bench/raw");
    let results_dir = repo_root.join("target/bench/results");
    fs::create_dir_all(&raw_dir)?;
    fs::create_dir_all(&results_dir)?;

    eprintln!("warmup={warmup} iters={iters} n_generate={N_GENERATE} prompt={PROMPT:?}");

    let runs = build_runs(&repo_root);
    let mut rows: Vec<(String, Aggregate)> = Vec::new();

    for spec in runs {
        if let Some(f) = &filter {
            if !spec.label.contains(f.as_str()) {
                continue;
            }
        }
        eprintln!(">>> {}", spec.label);
        let log_path = raw_dir.join(format!("{}.log", spec.label));
        let cmd = build_command(&spec, &repo_root, warmup, iters);
        if let Err(e) = run_with_monitors(cmd, &log_path) {
            eprintln!("    failed: {e}");
            continue;
        }
        let log_text = fs::read_to_string(&log_path)?;
        let mut a = match &spec.runtime {
            Runtime::Mynn {
                prefill_padded_for_aggr,
                ..
            } => aggregate_mynn(&log_text, *prefill_padded_for_aggr),
            Runtime::LlamaCpp { .. } => aggregate_llamacpp(&log_text, PROMPT_TOKENS),
            Runtime::OrtGenAI { .. } => {
                // ORT-GenAI consumes the full prompt in one prefill kernel; report
                // prefill_tok_s against the actual prompt length (no padding).
                aggregate_mynn(&log_text, PROMPT_TOKENS)
            }
        };
        let stem = log_path.with_extension("");
        a.peak_vram_mib = fs::read_to_string(stem.with_extension("vram.txt"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let rss_kb: u64 = fs::read_to_string(stem.with_extension("rss.txt"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        a.peak_rss_mib = rss_kb as f64 / 1024.0;
        rows.push((spec.label, a));
    }

    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out = String::new();
    out.push_str("| runtime | model | dtype | decode tok/s | prefill tok/s | TTFT (ms) | peak VRAM (MiB) | peak RAM (MiB) |\n");
    out.push_str("|---|---|---|---:|---:|---:|---:|---:|\n");
    for (label, a) in &rows {
        let (rt, model, dtype) = label_to_runtime_model_dtype(label);
        out.push_str(&fmt_row(rt, model, dtype, a));
        out.push('\n');
    }
    let summary_path = results_dir.join("SUMMARY.md");
    fs::write(&summary_path, &out)?;
    print!("{out}");
    eprintln!("wrote {}", summary_path.display());
    Ok(())
}
