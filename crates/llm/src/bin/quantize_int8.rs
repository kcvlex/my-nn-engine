use std::path::PathBuf;

use my_nn_engine_llm::quantize::quantize_safetensors_int8;
use my_nn_engine_llm::quantize::quantize_safetensors_int8_dir;
use my_nn_engine_llm::quantize::quantize_safetensors_int8_streaming;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "usage: quantize_int8 <input.safetensors|model_dir> <output.safetensors|output_dir>\n\
             - if output ends with .safetensors: single-file output (loads all tensors in RAM)\n\
             - otherwise: sharded output dir mirroring input layout (streaming, low RAM)"
        );
        std::process::exit(1);
    }
    let in_path = PathBuf::from(&args[1]);
    let out_path = PathBuf::from(&args[2]);

    let single_file_out = out_path
        .extension()
        .map(|e| e == "safetensors")
        .unwrap_or(false);

    let stats = if in_path.is_dir() && !single_file_out {
        quantize_safetensors_int8_streaming(&in_path, &out_path)
    } else if in_path.is_dir() {
        quantize_safetensors_int8_dir(&in_path, &out_path)
    } else {
        quantize_safetensors_int8(&in_path, &out_path)
    }
    .expect("quantize failed");
    println!(
        "quantized {} tensors, passed through {}",
        stats.quantized, stats.passthrough
    );
    println!("wrote {}", out_path.display());
}
