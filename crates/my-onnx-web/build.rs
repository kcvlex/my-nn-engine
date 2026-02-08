use std::process::Command;
use std::path::Path;

fn main() {
    let frontend_dir = Path::new("crates/my-onnx-web/frontend");
    let dist_dir = Path::new("crates/my-onnx-web/dist");

    println!("cargo:rerun-if-changed=crates/my-onnx-web/frontend/src");
    println!("cargo:rerun-if-changed=crates/my-onnx-web/frontend/package.json");
    println!("cargo:rerun-if-changed=crates/my-onnx-web/frontend/vite.config.ts");

    // Check if we're in the workspace root
    let in_workspace = Path::new("crates").exists();

    let actual_frontend_dir = if in_workspace {
        frontend_dir.to_path_buf()
    } else {
        // We're in the crate directory
        Path::new("frontend").to_path_buf()
    };

    let actual_node_modules = actual_frontend_dir.join("node_modules");

    // Check if node/npm is available
    let npm_check = Command::new("npm")
        .arg("--version")
        .output();

    if npm_check.is_err() {
        eprintln!("Warning: npm not found. Skipping frontend build.");
        eprintln!("The frontend will not be built. Install Node.js/npm to build the frontend.");
        return;
    }

    println!("cargo:warning=Building frontend...");

    // Install dependencies if node_modules doesn't exist
    if !actual_node_modules.exists() {
        println!("cargo:warning=Installing npm dependencies...");
        let npm_install = Command::new("npm")
            .arg("install")
            .current_dir(&actual_frontend_dir)
            .status();

        match npm_install {
            Ok(status) if status.success() => {
                println!("cargo:warning=npm install completed");
            }
            Ok(status) => {
                eprintln!("Warning: npm install failed with status: {}", status);
                eprintln!("You may need to run 'npm install' manually in {:?}", actual_frontend_dir);
                return;
            }
            Err(e) => {
                eprintln!("Warning: Failed to run npm install: {}", e);
                return;
            }
        }
    }

    // Build the frontend
    println!("cargo:warning=Running npm run build...");
    let npm_build = Command::new("npm")
        .arg("run")
        .arg("build")
        .current_dir(&actual_frontend_dir)
        .status();

    match npm_build {
        Ok(status) if status.success() => {
            println!("cargo:warning=Frontend build completed successfully");

            // Verify dist was created
            let actual_dist = if in_workspace {
                dist_dir.to_path_buf()
            } else {
                Path::new("dist").to_path_buf()
            };

            if actual_dist.exists() {
                println!("cargo:warning=Frontend assets available at {:?}", actual_dist);
            } else {
                eprintln!("Warning: dist directory not found after build");
            }
        }
        Ok(status) => {
            eprintln!("Warning: npm run build failed with status: {}", status);
            eprintln!("You may need to run 'npm run build' manually in {:?}", actual_frontend_dir);
        }
        Err(e) => {
            eprintln!("Warning: Failed to run npm run build: {}", e);
        }
    }
}
