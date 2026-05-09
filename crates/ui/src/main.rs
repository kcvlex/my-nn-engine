mod grpc_service;
mod llm;
mod models;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
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

#[derive(Parser)]
#[command(about = "my-nn-engine UI gRPC backend")]
struct Args {
    #[arg(
        long,
        value_name = "PATH",
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/hf"),
    )]
    models_root: PathBuf,

    #[arg(
        long,
        value_name = "PATH",
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/validated"),
    )]
    validated_root: PathBuf,

    #[arg(long, default_value_t = 50051, env = "PORT")]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    log::info!(
        "Validated models directory: {}",
        args.validated_root.display()
    );
    log::info!("LLM models root: {}", args.models_root.display());
    log::info!("Starting gRPC server on port {}", args.port);

    let model_registry = Arc::new(RwLock::new(ModelRegistry::new(&args.validated_root)));
    let chat_registry = Arc::new(ChatRegistry::new(&args.models_root));
    let service = OnnxInferenceServiceImpl::new(model_registry, chat_registry);

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::any())
        .allow_headers(AllowHeaders::any())
        .allow_methods(AllowMethods::any())
        .expose_headers(tower_http::cors::ExposeHeaders::list([
            http::header::HeaderName::from_static("grpc-status"),
            http::header::HeaderName::from_static("grpc-message"),
        ]));

    let addr = format!("0.0.0.0:{}", args.port).parse()?;
    log::info!("Serving on {}", addr);

    Server::builder()
        .accept_http1(true)
        .layer(cors)
        .add_service(tonic_web::enable(OnnxInferenceServiceServer::new(service)))
        .serve(addr)
        .await?;

    Ok(())
}
