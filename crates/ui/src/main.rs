mod grpc_service;
mod llm;
mod models;

use std::path::Path;
use std::sync::Arc;

use grpc_service::onnx_service::onnx_inference_service_server::OnnxInferenceServiceServer;
use grpc_service::OnnxInferenceServiceImpl;
use llm::ChatRegistry;
use models::ModelRegistry;
use tokio::sync::RwLock;
use tonic::transport::Server;
use tower_http::cors::AllowHeaders;
use tower_http::cors::AllowMethods;
use tower_http::cors::AllowOrigin;
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let workspace_models_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let models_dir = workspace_models_root.join("validated");
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "50051".to_string())
        .parse()
        .expect("PORT must be a valid u16");

    log::info!("Validated models directory: {}", models_dir.display());
    log::info!("LLM models root: {}", workspace_models_root.display());
    log::info!("Starting gRPC server on port {}", port);

    let model_registry = Arc::new(RwLock::new(ModelRegistry::new(&models_dir)));
    let chat_registry = Arc::new(ChatRegistry::new(&workspace_models_root));
    let service = OnnxInferenceServiceImpl::new(model_registry, chat_registry);

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::any())
        .allow_headers(AllowHeaders::any())
        .allow_methods(AllowMethods::any())
        .expose_headers(tower_http::cors::ExposeHeaders::list([
            http::header::HeaderName::from_static("grpc-status"),
            http::header::HeaderName::from_static("grpc-message"),
        ]));

    let addr = format!("0.0.0.0:{}", port).parse()?;
    log::info!("Serving on {}", addr);

    Server::builder()
        .accept_http1(true)
        .layer(cors)
        .add_service(tonic_web::enable(OnnxInferenceServiceServer::new(service)))
        .serve(addr)
        .await?;

    Ok(())
}
