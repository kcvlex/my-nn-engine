use std::path::PathBuf;

use my_nn_engine_llm::quantize::quantize_safetensors_int8;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: quantize_int8 <input.safetensors> <output.safetensors>");
        std::process::exit(1);
    }
    let in_path = PathBuf::from(&args[1]);
    let out_path = PathBuf::from(&args[2]);

    let stats = quantize_safetensors_int8(&in_path, &out_path).expect("quantize failed");
    println!(
        "quantized {} tensors, passed through {}",
        stats.quantized, stats.passthrough
    );
    println!("wrote {}", out_path.display());
}
