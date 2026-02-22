use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use clap::Parser;

const IMAGE_NAME: &str = "onnx-test-tools";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    mode: Mode,

    #[arg(long)]
    model_path: PathBuf,

    #[arg(long, num_args(1..))]
    input_paths: Vec<PathBuf>,

    #[arg(long)]
    output_node_name: Option<String>,

    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, num_args(1..))]
    test_command: Option<Vec<String>>,
}

#[derive(clap::ValueEnum, Clone)]
enum Mode {
    Extract,
    BinarySearch,
}

struct Extract {
    project_root: PathBuf,
    model_path: PathBuf,
    input_paths: Vec<PathBuf>,
    output_node_name: String,
    output_dir: PathBuf,
}

struct BinarySearch {
    project_root: PathBuf,
    model_path: PathBuf,
    input_paths: Vec<PathBuf>,
    test_command: Vec<String>,
    output_dir: PathBuf,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let project_root = get_project_root()?;
    let dockerfile_dir = project_root.join("crates").join("onnx-test-tools");

    if !image_exists()? {
        build_image(&dockerfile_dir)?;
    }

    match args.mode {
        Mode::Extract => {
            let config = Extract {
                project_root,
                output_node_name: args
                    .output_node_name
                    .ok_or("--output-node-name is required for extract mode")?,
                model_path: args.model_path,
                input_paths: args.input_paths,
                output_dir: args.output_dir,
            };
            run_extraction(config)?;
        }
        Mode::BinarySearch => {
            let config = BinarySearch {
                project_root,
                test_command: args
                    .test_command
                    .ok_or("--test-command is required for binary-search mode")?,
                model_path: args.model_path,
                input_paths: args.input_paths,
                output_dir: args.output_dir,
            };
            run_binary_search(config)?;
        }
    }

    info("Done!");
    Ok(())
}

fn run_extraction(config: Extract) -> Result<(), Box<dyn std::error::Error>> {
    info("Running extraction...");

    let mut container_args = vec![
        "--model".to_string(),
        config.model_path.to_string_lossy().to_string(),
        "--outputs".to_string(),
        config.output_node_name,
        "--output-dir".to_string(),
        config.output_dir.to_string_lossy().to_string(),
    ];

    container_args.push("--inputs".to_string());
    for input in &config.input_paths {
        container_args.push(input.to_string_lossy().to_string());
    }

    let status = Command::new("podman")
        .args([
            "run",
            "--rm",
            "--userns=keep-id",
            "-v",
            &format!(
                "{}:/workspace",
                config
                    .project_root
                    .to_str()
                    .ok_or("Invalid project root path")?
            ),
            "-w",
            "/workspace",
            IMAGE_NAME,
        ])
        .args(&container_args)
        .status()?;

    if !status.success() {
        return Err("Extraction failed".into());
    }

    Ok(())
}

fn run_binary_search(config: BinarySearch) -> Result<(), Box<dyn std::error::Error>> {
    info("Running binary search to find problematic node...");

    let mut container_args = vec![
        "/usr/local/bin/binary_search.py".to_string(),
        "--model".to_string(),
        config.model_path.to_string_lossy().to_string(),
        "--temp-dir".to_string(),
        config.output_dir.to_string_lossy().to_string(),
    ];

    container_args.push("--inputs".to_string());
    for input in &config.input_paths {
        container_args.push(input.to_string_lossy().to_string());
    }

    container_args.push("--test-command".to_string());
    for cmd in &config.test_command {
        container_args.push(cmd.clone());
    }

    let status = Command::new("podman")
        .args([
            "run",
            "--rm",
            "--userns=keep-id",
            "--entrypoint",
            "python",
            "-v",
            &format!(
                "{}:/workspace",
                config
                    .project_root
                    .to_str()
                    .ok_or("Invalid project root path")?
            ),
            "-w",
            "/workspace",
            IMAGE_NAME,
        ])
        .args(&container_args)
        .status()?;

    if !status.success() {
        return Err("Binary search failed or found a problematic node".into());
    }

    Ok(())
}

fn get_project_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
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

    env::current_dir().map_err(Into::into)
}

fn image_exists() -> Result<bool, Box<dyn std::error::Error>> {
    let status = Command::new("podman")
        .args(["image", "inspect", IMAGE_NAME])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;

    Ok(status.success())
}

fn build_image(dockerfile_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    info("Building Podman image...");

    let status = Command::new("podman")
        .args(["build", "-t", IMAGE_NAME])
        .arg(dockerfile_dir)
        .status()?;

    if !status.success() {
        return Err("Failed to build Podman image".into());
    }

    info("Podman image built successfully");
    Ok(())
}

fn info(msg: &str) {
    eprintln!("\x1b[32m[INFO]\x1b[0m {}", msg);
}
