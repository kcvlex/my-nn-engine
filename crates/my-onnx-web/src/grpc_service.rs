use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use crate::models::{ModelId, ModelRegistry};
use my_onnx::options::Target as MyOnnxTarget;
use my_onnx::tensor::Tensor;

// Include generated protobuf code
pub mod onnx_service {
    tonic::include_proto!("onnx_service");
}

use onnx_service::onnx_inference_service_server::OnnxInferenceService;
use onnx_service::*;

pub struct OnnxInferenceServiceImpl {
    registry: Arc<RwLock<ModelRegistry>>,
}

impl OnnxInferenceServiceImpl {
    pub fn new(registry: Arc<RwLock<ModelRegistry>>) -> Self {
        Self { registry }
    }

    /// Convert protobuf TensorData to internal Tensor
    fn tensor_data_to_tensor(data: &TensorData) -> Result<Tensor, Status> {
        let dims: Vec<usize> = data.dims.iter().map(|&d| d as usize).collect();

        // Create ndarray from data and dims
        let array = ndarray::Array::from_shape_vec(dims, data.data.clone())
            .map_err(|e| Status::invalid_argument(format!("Invalid tensor shape: {:?}", e)))?
            .into_dyn();

        // Convert to Tensor
        Tensor::try_from(array)
            .map_err(|e| Status::invalid_argument(format!("Failed to create tensor: {:?}", e)))
    }

    /// Convert internal Tensor to protobuf TensorData
    fn tensor_to_tensor_data(name: String, tensor: &Tensor) -> TensorData {
        let dims: Vec<i64> = tensor.dims.inner().iter().map(|&d| d as i64).collect();

        // Extract data as f64 vector
        let data = match &tensor.data {
            my_onnx::tensor::data::TensorData::Float(_, v) => v.clone(),
            my_onnx::tensor::data::TensorData::SInt(_, v) => {
                v.iter().map(|&x| x as f64).collect()
            }
            my_onnx::tensor::data::TensorData::UInt(_, v) => {
                v.iter().map(|&x| x as f64).collect()
            }
        };

        let dtype = format!("{:?}", tensor.data.elem_type());

        TensorData {
            name,
            dims,
            dtype,
            data,
        }
    }

    /// Convert protobuf ModelId to internal ModelId
    fn proto_model_id_to_model_id(proto_id: i32) -> Result<ModelId, Status> {
        match ModelId::try_from(proto_id) {
            Ok(model_id::Mnist) => Ok(ModelId::Mnist),
            Ok(model_id::Resnet) => Ok(ModelId::ResNet),
            Ok(model_id::Yolo) => Ok(ModelId::Yolo),
            Ok(model_id::Bert) => Ok(ModelId::Bert),
            Ok(model_id::Gpt2) => Ok(ModelId::Gpt2),
            Err(_) => Err(Status::invalid_argument(format!(
                "Invalid model ID: {}",
                proto_id
            ))),
        }
    }

    /// Convert protobuf Backend to internal Target
    fn proto_backend_to_target(proto_backend: i32) -> Result<MyOnnxTarget, Status> {
        match Backend::try_from(proto_backend) {
            Ok(Backend::Cpu) => Ok(MyOnnxTarget::CPU),
            Ok(Backend::Cuda) => Ok(MyOnnxTarget::CUDA),
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

        // Convert protobuf types to internal types
        let model_id = Self::proto_model_id_to_model_id(req.model_id)?;
        let target = Self::proto_backend_to_target(req.backend)?;

        // Get the model from registry
        let registry = self.registry.read().await;
        let session = registry
            .get(model_id, target)
            .ok_or_else(|| {
                Status::not_found(format!(
                    "Model {} not loaded for backend {:?}",
                    model_id.display_name(),
                    target
                ))
            })?;

        // Convert input tensor
        let input_data = req
            .input_data
            .ok_or_else(|| Status::invalid_argument("Missing input_data"))?;

        let input_tensor = Self::tensor_data_to_tensor(&input_data)?;

        // Run inference with timing
        let start = std::time::Instant::now();
        let outputs = session
            .run(&[input_tensor])
            .map_err(|e| Status::internal(format!("Inference failed: {:?}", e)))?;
        let inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Convert output tensor (assuming single output)
        let output_tensor = outputs
            .first()
            .ok_or_else(|| Status::internal("No output from inference"))?;

        let output_data = Self::tensor_to_tensor_data("output".to_string(), output_tensor);

        Ok(Response::new(InferenceResponse {
            output_data: Some(output_data),
            inference_time_ms,
        }))
    }
}
