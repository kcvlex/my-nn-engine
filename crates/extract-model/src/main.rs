use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

const IMAGE_NAME: &str = "my-onnx-extract";

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();

    // Check for --help
    if args.is_empty() || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_usage();
        return Ok(());
    }

    // Determine project root
    let project_root = get_project_root()?;
    let dockerfile_dir = project_root.join("crates").join("extract-model");

    // Check for --rebuild flag
    let rebuild = args.contains(&"--rebuild".to_string());
    let extract_args: Vec<String> = args.into_iter().filter(|arg| arg != "--rebuild").collect();

    // Detect Docker command (with or without sudo)
    let docker_cmd = detect_docker()?;
    info(&format!("Using Docker command: {}", docker_cmd.join(" ")));

    // Build image if needed
    if rebuild || !image_exists(&docker_cmd)? {
        build_image(&docker_cmd, &dockerfile_dir)?;
    } else {
        info("Using existing Docker image (use --rebuild to rebuild)");
    }

    // Run extraction
    run_extraction(&docker_cmd, &project_root, &extract_args)?;

    info("Done!");
    Ok(())
}

fn get_project_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Try to find Cargo.toml by walking up the directory tree
    let mut current = env::current_exe()?;
    loop {
        current.pop();
        if current.join("Cargo.toml").exists() {
            return Ok(current);
        }
        if !current.pop() {
            break;
        }
    }

    // Fallback: assume we're running from project root
    env::current_dir().map_err(Into::into)
}

fn detect_docker() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Try without sudo first
    let status = Command::new("docker")
        .arg("ps")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if status.is_ok() && status.unwrap().success() {
        return Ok(vec!["docker".to_string()]);
    }

    // Try with sudo
    let status = Command::new("sudo")
        .args(&["docker", "ps"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if status.is_ok() && status.unwrap().success() {
        warn("Using sudo for Docker commands");
        return Ok(vec!["sudo".to_string(), "docker".to_string()]);
    }

    Err("Cannot access Docker. Please ensure Docker is installed and running.".into())
}

fn image_exists(docker_cmd: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    let status = Command::new(&docker_cmd[0])
        .args(&docker_cmd[1..])
        .args(&["image", "inspect", IMAGE_NAME])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;

    Ok(status.success())
}

fn build_image(
    docker_cmd: &[String],
    dockerfile_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    info("Building Docker image...");

    let status = Command::new(&docker_cmd[0])
        .args(&docker_cmd[1..])
        .args(&["build", "-t", IMAGE_NAME])
        .arg(dockerfile_dir)
        .status()?;

    if !status.success() {
        return Err("Failed to build Docker image".into());
    }

    info("Docker image built successfully");
    Ok(())
}

fn run_extraction(
    docker_cmd: &[String],
    project_root: &PathBuf,
    args: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    info("Running extraction...");

    let project_root_str = project_root.to_str().ok_or("Invalid project root path")?;

    let status = Command::new(&docker_cmd[0])
        .args(&docker_cmd[1..])
        .args(&[
            "run",
            "--rm",
            "-v",
            &format!("{}:/workspace", project_root_str),
            "-w",
            "/workspace",
            IMAGE_NAME,
        ])
        .args(args)
        .status()?;

    if !status.success() {
        return Err("Extraction failed".into());
    }

    Ok(())
}

fn info(msg: &str) {
    eprintln!("\x1b[32m[INFO]\x1b[0m {}", msg);
}

fn warn(msg: &str) {
    eprintln!("\x1b[33m[WARN]\x1b[0m {}", msg);
}

fn print_usage() {
    println!(
        r#"Usage: extract-model [--rebuild] --model MODEL --inputs INPUT... --outputs OUTPUT... --output-dir DIR [OPTIONS]

Extract a subgraph from an ONNX model and generate test data.

Required Arguments:
  --model PATH              Path to the input ONNX model (relative to project root)
  --inputs PATH...          Paths to input tensor .pb files (can use wildcards)
  --outputs NAME...         Names of output nodes to extract up to
  --output-dir DIR          Directory to save extracted model and test data

Optional Arguments:
  --input-names NAME...     Names of input nodes (if different from original)
  --no-check                Skip model validation check
  --rebuild                 Rebuild the Docker image before running

Examples:
  # Extract yolov4 subgraph
  extract-model \
    --model models/validated/yolov4/yolov4.onnx \
    --inputs models/validated/yolov4/test_data_set_0/input_0.pb \
    --outputs "lambda_5/add:0" \
    --output-dir models/extracted/yolov4/until_lambda_5_add

  # Extract bertsquad subgraph with multiple inputs
  extract-model \
    --model models/validated/bertsquad-12/bertsquad-12.onnx \
    --inputs models/validated/bertsquad-12/test_data_set_0/input_*.pb \
    --outputs "bert/encoder/Cast:0" \
    --output-dir models/extracted/bertsquad-12/until_encoder_cast

Note: This tool automatically uses sudo for Docker if needed.
"#
    );
}
