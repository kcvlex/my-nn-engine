use std::env;
use std::os::unix::fs::PermissionsExt;
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
    output_dir: Option<PathBuf>,

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
    work_dir: tempfile::TempDir,
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
                output_dir: args
                    .output_dir
                    .ok_or("--output-dir is required for extract mode")?,
            };
            run_extraction(config)?;
        }
        Mode::BinarySearch => {
            let work_dir = tempfile::TempDir::new_in(&project_root)?;
            std::fs::set_permissions(work_dir.path(), std::fs::Permissions::from_mode(0o777))?;
            let config = BinarySearch {
                project_root,
                test_command: args
                    .test_command
                    .ok_or("--test-command is required for binary-search mode")?,
                model_path: args.model_path,
                input_paths: args.input_paths,
                work_dir,
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

/// Run a container command and capture its stdout.
fn container_output(
    project_root: &PathBuf,
    script: &str,
    args: &[String],
) -> Result<String, Box<dyn std::error::Error>> {
    let volume = format!(
        "{}:/workspace",
        project_root.to_str().ok_or("Invalid project root path")?
    );

    let output = Command::new("podman")
        .args([
            "run",
            "--rm",
            "--userns=keep-id",
            "--entrypoint",
            "python",
            "-v",
            &volume,
            "-w",
            "/workspace",
            IMAGE_NAME,
            script,
        ])
        .args(args)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Container command failed: {}", stderr).into());
    }

    let stdout = String::from_utf8(output.stdout)?.trim().to_string();
    Ok(stdout)
}

fn run_binary_search(config: BinarySearch) -> Result<(), Box<dyn std::error::Error>> {
    info("Running binary search to find problematic node...");

    let work_dir_rel = config
        .work_dir
        .path()
        .strip_prefix(&config.project_root)?
        .to_path_buf();

    // Step 1: Get total node count from container
    let total_nodes: usize = {
        let args = vec![
            "list-nodes".to_string(),
            "--model".to_string(),
            config.model_path.to_string_lossy().to_string(),
        ];
        let out = container_output(
            &config.project_root,
            "/usr/local/bin/binary_search.py",
            &args,
        )?;
        out.parse::<usize>()
            .map_err(|e| format!("Failed to parse node count '{}': {}", out, e))?
    };

    eprintln!("Model: {}", config.model_path.display());
    eprintln!("Total nodes: {}", total_nodes);
    eprintln!();

    if total_nodes == 0 {
        info("Model has no nodes.");
        return Ok(());
    }

    // Step 2: Binary search loop
    let mut left: usize = 0;
    let mut right: usize = total_nodes - 1;
    let mut first_fail: Option<(usize, String)> = None;

    while left <= right {
        let mid = (left + right) / 2;

        // 2b. Call container: extract-node
        let node_dir_rel = work_dir_rel.join(format!("node_{}", mid));
        let mut extract_args = vec![
            "extract-node".to_string(),
            "--model".to_string(),
            config.model_path.to_string_lossy().to_string(),
            "--node-index".to_string(),
            mid.to_string(),
            "--output-dir".to_string(),
            node_dir_rel.to_string_lossy().to_string(),
        ];
        extract_args.push("--inputs".to_string());
        for input in &config.input_paths {
            extract_args.push(input.to_string_lossy().to_string());
        }

        eprint!("Testing node [{}/{}] ", mid, total_nodes - 1,);

        let node_info = match container_output(
            &config.project_root,
            "/usr/local/bin/binary_search.py",
            &extract_args,
        ) {
            Ok(info) => {
                eprintln!("{}", info);
                info
            }
            Err(e) => {
                eprintln!("extraction failed: {}", e);
                // Treat extraction failure as a test failure
                first_fail = Some((mid, format!("node_{} (extraction failed)", mid)));
                let Some(new_right) = mid.checked_sub(1) else {
                    break;
                };
                right = new_right;
                continue;
            }
        };

        // 2c. Run test command on the host
        let extracted_model_dir = config.project_root.join(&node_dir_rel);
        let status = Command::new(&config.test_command[0])
            .args(&config.test_command[1..])
            .env("EXTRACTED_MODEL_DIR", &extracted_model_dir)
            .status()?;

        if status.success() {
            eprintln!("  PASS");
            left = mid + 1;
        } else {
            eprintln!("  FAIL");
            first_fail = Some((mid, node_info));
            let Some(new_right) = mid.checked_sub(1) else {
                break;
            };
            right = new_right;
        }

        eprintln!();
    }

    // Step 3: Print results
    eprintln!();
    if let Some((index, node_info)) = first_fail {
        eprintln!("{}", "=".repeat(80));
        eprintln!("FOUND: First failing node");
        eprintln!("{}", "=".repeat(80));
        eprintln!("Index:    {}", index);
        eprintln!("Node:     {}", node_info);
        eprintln!(
            "Extracted: {}",
            config
                .work_dir
                .path()
                .join(format!("node_{}", index))
                .display()
        );
        return Err("Binary search found a failing node".into());
    } else {
        eprintln!("{}", "=".repeat(80));
        eprintln!("All nodes pass! No failing node found.");
        eprintln!("{}", "=".repeat(80));
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
