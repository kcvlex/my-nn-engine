use std::sync::Arc;
use axum::{routing::{get, post}, Router};
use tower_http::cors::CorsLayer;

use my_onnx_web::{
    api::{self, AppState},
    SessionCache,
};

#[tokio::main]
async fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Create cache and upload directories
    let cache_dir = std::path::PathBuf::from("./cache");
    let upload_dir = std::path::PathBuf::from("./uploads");
    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    std::fs::create_dir_all(&upload_dir).expect("Failed to create upload directory");

    // Initialize session cache
    let cache = Arc::new(SessionCache::new(cache_dir));

    // Create application state
    let state = AppState {
        cache,
        upload_dir,
    };

    // Build the application router
    let app = Router::new()
        .route("/", get(api::serve_ui))
        .route("/models/upload", post(api::upload_model))
        .route("/models/:id/infer", post(api::run_inference))
        .route("/models", get(api::list_models))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start the server
    let addr = "0.0.0.0:3000";
    log::info!("Starting web server on {}", addr);
    log::info!("Open http://localhost:3000 in your browser");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app)
        .await
        .expect("Failed to start server");
}
