use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use my_nn_engine::options::PrefetchPolicy;
use my_nn_engine::options::Target as MyOnnxTarget;
use my_nn_engine::tensor::Tensor;
use prost::Message;
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

use crate::llm::ChatRegistry;
use crate::llm::ChatStreamEvent;
use crate::models::ModelId;
use crate::models::ModelRegistry;

// Include generated protobuf code
pub mod onnx_service {
    tonic::include_proto!("onnx_service");
}

pub mod onnx {
    tonic::include_proto!("onnx");
}

use onnx_service::chat_response::Event as ChatEvent;
use onnx_service::onnx_inference_service_server::OnnxInferenceService;
use onnx_service::Backend as ProtoBackend;
use onnx_service::ChatChunk;
use onnx_service::ChatDone;
use onnx_service::ChatRequest;
use onnx_service::ChatResponse;
use onnx_service::CreateChatSessionRequest;
use onnx_service::CreateChatSessionResponse;
use onnx_service::DestroyChatSessionRequest;
use onnx_service::DestroyChatSessionResponse;
use onnx_service::GetInitializerRequest;
use onnx_service::GetInitializerResponse;
use onnx_service::InferenceRequest;
use onnx_service::InferenceResponse;
use onnx_service::ListLlmModelsRequest;
use onnx_service::ListLlmModelsResponse;
use onnx_service::LlmModelInfo;
use onnx_service::ModelId as ProtoModelId;
use onnx_service::WarmUpRequest;
use onnx_service::WarmUpResponse;

pub struct OnnxInferenceServiceImpl {
    registry: Arc<RwLock<ModelRegistry>>,
    chat_registry: Arc<ChatRegistry>,
}

impl OnnxInferenceServiceImpl {
    pub fn new(registry: Arc<RwLock<ModelRegistry>>, chat_registry: Arc<ChatRegistry>) -> Self {
        Self {
            registry,
            chat_registry,
        }
    }

    fn proto_model_id_to_model_id(proto_id: i32) -> Result<ModelId, Status> {
        match ProtoModelId::try_from(proto_id) {
            Ok(ProtoModelId::Mnist) => Ok(ModelId::Mnist),
            Ok(ProtoModelId::Resnet) => Ok(ModelId::ResNet),
            Ok(ProtoModelId::Resnet152) => Ok(ModelId::ResNet152),
            Ok(ProtoModelId::Mobilenetv2) => Ok(ModelId::MobileNetV2),
            Ok(ProtoModelId::EfficientnetLite4) => Ok(ModelId::EfficientNetLite4),
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
            Ok(ProtoBackend::Cuda) => Ok(MyOnnxTarget::CUDA(PrefetchPolicy::Disabled)),
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
            .get_or_load(model_id, target)
            .map_err(|e| Status::internal(e))?;

        // Run inference
        let start = std::time::Instant::now();
        let outputs = session
            .lock()
            .map_err(|e| Status::internal(format!("Session lock failed: {:?}", e)))?
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

    async fn warm_up(
        &self,
        request: Request<WarmUpRequest>,
    ) -> Result<Response<WarmUpResponse>, Status> {
        let req = request.into_inner();
        let model_id = Self::proto_model_id_to_model_id(req.model_id)?;
        let target = Self::proto_backend_to_target(req.backend)?;

        let mut registry = self.registry.write().await;
        let already_loaded = registry.is_loaded(model_id, target);
        let start = std::time::Instant::now();
        registry
            .warm_up(model_id, target)
            .map_err(|e| Status::internal(e))?;
        let build_time_ms = if already_loaded {
            0.0
        } else {
            start.elapsed().as_secs_f64() * 1000.0
        };

        Ok(Response::new(WarmUpResponse { build_time_ms }))
    }

    async fn get_initializer(
        &self,
        request: Request<GetInitializerRequest>,
    ) -> Result<Response<GetInitializerResponse>, Status> {
        let req = request.into_inner();

        if req.name.is_empty() ||
            !req.name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '.' || c == '_') ||
            req.name.contains("..")
        {
            return Err(Status::invalid_argument(
                "Initializer name must be non-empty and contain only alphanumeric characters, dots, or underscores, and must not contain '..'",
            ));
        }

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

    async fn list_llm_models(
        &self,
        _request: Request<ListLlmModelsRequest>,
    ) -> Result<Response<ListLlmModelsResponse>, Status> {
        let models = self
            .chat_registry
            .list()
            .into_iter()
            .map(|dir_name| LlmModelInfo { dir_name })
            .collect();
        Ok(Response::new(ListLlmModelsResponse { models }))
    }

    async fn create_chat_session(
        &self,
        request: Request<CreateChatSessionRequest>,
    ) -> Result<Response<CreateChatSessionResponse>, Status> {
        let req = request.into_inner();
        let target = Self::proto_backend_to_target(req.backend)?;
        let model_dir = req.model_dir;

        let chat_registry = Arc::clone(&self.chat_registry);
        let (session_id, build_time) =
            tokio::task::spawn_blocking(move || chat_registry.create(&model_dir, target))
                .await
                .map_err(|e| Status::internal(format!("join: {e}")))?
                .map_err(|e| Status::internal(format!("{e}")))?;

        Ok(Response::new(CreateChatSessionResponse {
            session_id,
            build_time_ms: build_time.as_secs_f64() * 1000.0,
        }))
    }

    type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatResponse, Status>> + Send + 'static>>;

    async fn chat(
        &self,
        request: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStream>, Status> {
        let req = request.into_inner();
        let chat_registry = Arc::clone(&self.chat_registry);

        // Bounded enough to avoid runaway memory if the client is slow but
        // generous enough that the generator rarely blocks on emit. The
        // generator runs in spawn_blocking so blocking_send is safe.
        let (tx, rx) = mpsc::channel::<Result<ChatResponse, Status>>(64);

        tokio::task::spawn_blocking(move || {
            let result = chat_registry.chat_stream(
                &req.session_id,
                req.user_message,
                req.max_tokens,
                &mut |event| {
                    let resp = match event {
                        ChatStreamEvent::Chunk { delta } => ChatResponse {
                            event: Some(ChatEvent::Chunk(ChatChunk { delta })),
                        },
                        ChatStreamEvent::Done(d) => ChatResponse {
                            event: Some(ChatEvent::Done(ChatDone {
                                tokens_generated: d.tokens_generated,
                                generation_time_ms: d.generation_time.as_secs_f64() * 1000.0,
                                eos_emitted: d.eos_emitted,
                            })),
                        },
                    };
                    let _ = tx.blocking_send(Ok(resp));
                },
            );
            if let Err(e) = result {
                let _ = tx.blocking_send(Err(Status::internal(format!("{e}"))));
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn destroy_chat_session(
        &self,
        request: Request<DestroyChatSessionRequest>,
    ) -> Result<Response<DestroyChatSessionResponse>, Status> {
        let req = request.into_inner();
        self.chat_registry.destroy(&req.session_id);
        Ok(Response::new(DestroyChatSessionResponse {}))
    }
}
