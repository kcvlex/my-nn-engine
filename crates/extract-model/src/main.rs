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

    // Check for --binary-search flag
    let binary_search = args.contains(&"--binary-search".to_string());

    // Check for --rebuild flag
    let rebuild = args.contains(&"--rebuild".to_string());
    let extract_args: Vec<String> = args
        .into_iter()
        .filter(|arg| arg != "--rebuild" && arg != "--binary-search")
        .collect();

    // Detect Podman command (with or without sudo)
    let docker_cmd = detect_docker()?;
    info(&format!("Using Podman command: {}", docker_cmd.join(" ")));

    // Build image if needed
    if rebuild || !image_exists(&docker_cmd)? {
        build_image(&docker_cmd, &dockerfile_dir)?;
    } else {
        info("Using existing Podman image (use --rebuild to rebuild)");
    }

    // Run extraction or binary search
    if binary_search {
        run_binary_search(&docker_cmd, &project_root, &extract_args)?;
    } else {
        run_extraction(&docker_cmd, &project_root, &extract_args)?;
    }

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
    // Try podman first (rootless by default)
    let status = Command::new("podman")
        .arg("ps")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if status.is_ok() && status.unwrap().success() {
        return Ok(vec!["podman".to_string()]);
    }

    // Fallback to podman with sudo
    let status = Command::new("sudo")
        .args(&["podman", "ps"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if status.is_ok() && status.unwrap().success() {
        warn("Using sudo for Podman commands");
        return Ok(vec!["sudo".to_string(), "podman".to_string()]);
    }

    Err("Cannot access Podman. Please ensure Podman is installed and running.".into())
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
    info("Building Podman image...");

    let status = Command::new(&docker_cmd[0])
        .args(&docker_cmd[1..])
        .args(&["build", "-t", IMAGE_NAME])
        .arg(dockerfile_dir)
        .status()?;

    if !status.success() {
        return Err("Failed to build Podman image".into());
    }

    info("Podman image built successfully");
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
            "--userns=keep-id", // Map container user to host user (Podman rootless)
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

fn run_binary_search(
    docker_cmd: &[String],
    project_root: &PathBuf,
    args: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    info("Running binary search to find problematic node...");

    let project_root_str = project_root.to_str().ok_or("Invalid project root path")?;

    let status = Command::new(&docker_cmd[0])
        .args(&docker_cmd[1..])
        .args(&[
            "run",
            "--rm",
            "--userns=keep-id", // Map container user to host user (Podman rootless)
            "-v",
            &format!("{}:/workspace", project_root_str),
            "-w",
            "/workspace",
            IMAGE_NAME,
            "python",
            "/usr/local/bin/binary_search.py",
        ])
        .args(args)
        .status()?;

    if !status.success() {
        return Err("Binary search failed or found a problematic node".into());
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
        r#"Usage: extract-model [OPTIONS] COMMAND_ARGS

Extract a subgraph from an ONNX model and generate test data, or use binary
search to find the node that produces incorrect results.

=== EXTRACTION MODE (default) ===

Required Arguments:
  --model PATH              Path to the input ONNX model (relative to project root)
  --inputs PATH...          Paths to input tensor .pb files (can use wildcards)
  --outputs NAME...         Names of output nodes to extract up to
  --output-dir DIR          Directory to save extracted model and test data

Optional Arguments:
  --input-names NAME...     Names of input nodes (if different from original)
  --no-check                Skip model validation check
  --rebuild                 Rebuild the Podman image before running

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

=== BINARY SEARCH MODE ===

Use --binary-search flag to automatically find the first node that produces
incorrect results by performing binary search on all nodes in the model.

Required Arguments:
  --binary-search           Enable binary search mode
  --model PATH              Path to the input ONNX model (relative to project root)
  --inputs PATH...          Paths to input tensor .pb files (can use wildcards)
  --test-command CMD...     Command to run tests on extracted models

Optional Arguments:
  --temp-dir DIR            Temporary directory for extracted models (default: /tmp/binary_search_nodes)
  --rebuild                 Rebuild the Podman image before running

Example:
  # Find the problematic node in bertsquad-12
  extract-model --binary-search \
    --model models/validated/bertsquad-12/bertsquad-12.onnx \
    --inputs models/validated/bertsquad-12/test_data_set_0/input_*.pb \
    --test-command cargo test --test extracted_models -- --nocapture

Note: This tool automatically uses sudo for Podman if needed.
"#
    );
}
