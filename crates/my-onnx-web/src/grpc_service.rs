use std::sync::Arc;

use my_onnx::options::Target as MyOnnxTarget;
use my_onnx::tensor::Tensor;
use prost::Message;
use tokio::sync::RwLock;
use tonic::Request;
use tonic::Response;
use tonic::Status;

use crate::models::ModelId;
use crate::models::ModelRegistry;

// Include generated protobuf code
pub mod onnx_service {
    tonic::include_proto!("onnx_service");
}

pub mod onnx {
    tonic::include_proto!("onnx");
}

use onnx_service::onnx_inference_service_server::OnnxInferenceService;
use onnx_service::Backend as ProtoBackend;
use onnx_service::GetInitializerRequest;
use onnx_service::GetInitializerResponse;
use onnx_service::InferenceRequest;
use onnx_service::InferenceResponse;
use onnx_service::ModelId as ProtoModelId;

pub struct OnnxInferenceServiceImpl {
    registry: Arc<RwLock<ModelRegistry>>,
}

impl OnnxInferenceServiceImpl {
    pub fn new(registry: Arc<RwLock<ModelRegistry>>) -> Self {
        Self { registry }
    }

    fn proto_model_id_to_model_id(proto_id: i32) -> Result<ModelId, Status> {
        match ProtoModelId::try_from(proto_id) {
            Ok(ProtoModelId::Mnist) => Ok(ModelId::Mnist),
            Ok(ProtoModelId::Resnet) => Ok(ModelId::ResNet),
            Ok(ProtoModelId::Yolo) => Ok(ModelId::Yolo),
            Ok(ProtoModelId::Bert) => Ok(ModelId::Bert),
            Ok(ProtoModelId::Gpt2) => Ok(ModelId::Gpt2),
            Err(_) => Err(Status::invalid_argument(format!(
                "Invalid model ID: {}",
                proto_id
            ))),
        }
    }

    fn proto_backend_to_target(proto_backend: i32) -> Result<MyOnnxTarget, Status> {
        match ProtoBackend::try_from(proto_backend) {
            Ok(ProtoBackend::Cpu) => Ok(MyOnnxTarget::CPU),
            Ok(ProtoBackend::Cuda) => Ok(MyOnnxTarget::CUDA),
            Err(_) => Err(Status::invalid_argument(format!(
                "Invalid backend: {}",
                proto_backend
            ))),
        }
    }
}

#[tonic::async_trait]
impl OnnxInferenceService for OnnxInferenceServiceImpl {
    async fn run_inference(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let req = request.into_inner();

        let model_id = Self::proto_model_id_to_model_id(req.model_id)?;
        let target = Self::proto_backend_to_target(req.backend)?;

        // Decode input TensorProtos into Tensors
        let input_tensors: Vec<Tensor> = req
            .inputs
            .iter()
            .map(|proto| {
                let bytes = proto.encode_to_vec();
                Tensor::from_proto_bytes(&bytes)
                    .map_err(|e| Status::invalid_argument(format!("Invalid input tensor: {:?}", e)))
            })
            .collect::<Result<_, _>>()?;

        // Get or load the model from registry
        let mut registry = self.registry.write().await;
        let session = registry
            .get_or_load(model_id, target, &input_tensors)
            .map_err(|e| Status::internal(e))?;

        // Run inference
        let start = std::time::Instant::now();
        let outputs = session
            .run(&input_tensors)
            .map_err(|e| Status::internal(format!("Inference failed: {:?}", e)))?;
        let inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Encode output Tensors back to TensorProtos
        let output_protos: Vec<onnx::TensorProto> = outputs
            .iter()
            .map(|tensor| {
                let bytes = tensor.to_proto_bytes();
                onnx::TensorProto::decode(bytes.as_slice())
                    .map_err(|e| Status::internal(format!("Failed to encode output: {:?}", e)))
            })
            .collect::<Result<_, _>>()?;

        Ok(Response::new(InferenceResponse {
            outputs: output_protos,
            inference_time_ms,
        }))
    }

    async fn get_initializer(
        &self,
        request: Request<GetInitializerRequest>,
    ) -> Result<Response<GetInitializerResponse>, Status> {
        let req = request.into_inner();
        let model_id = Self::proto_model_id_to_model_id(req.model_id)?;

        let registry = self.registry.read().await;
        let model_path = registry.model_path(model_id);

        let model_bytes = std::fs::read(&model_path)
            .map_err(|e| Status::not_found(format!("Failed to read model file: {}", e)))?;
        let model_proto = onnx::ModelProto::decode(model_bytes.as_slice())
            .map_err(|e| Status::internal(format!("Failed to decode model: {}", e)))?;
        let graph = model_proto
            .graph
            .ok_or_else(|| Status::internal("Model has no graph"))?;

        let tensor = graph
            .initializer
            .into_iter()
            .find(|t| t.name == req.name)
            .ok_or_else(|| {
                Status::not_found(format!(
                    "Initializer '{}' not found in model {}",
                    req.name,
                    model_id.display_name()
                ))
            })?;

        Ok(Response::new(GetInitializerResponse {
            tensor: Some(tensor),
        }))
    }
}
