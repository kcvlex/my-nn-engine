use std::sync::Arc;
use axum::{
    body::Bytes,
    extract::{Path, State, Multipart},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    Json,
};
use serde_derive::{Deserialize, Serialize};
use serde_json::json;

use my_onnx::options::{Options, Target};
use my_onnx::tensor::Tensor;
use my_onnx::onnx::load::ModelLoadError;
use crate::cache::SessionCache;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub cache: Arc<SessionCache>,
    pub upload_dir: std::path::PathBuf,
}

/// API error response
#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(json!({
            "error": message
        }));

        (status, body).into_response()
    }
}

impl From<my_onnx::session::SessionError> for ApiError {
    fn from(err: my_onnx::session::SessionError) -> Self {
        ApiError::InternalError(format!("{:?}", err))
    }
}

impl From<ModelLoadError> for ApiError {
    fn from(err: ModelLoadError) -> Self {
        ApiError::BadRequest(format!("Tensor decode error: {:?}", err))
    }
}

/// Request body for uploading a model
#[derive(Debug, Deserialize)]
pub struct UploadRequest {
    pub target: Option<String>,
    pub optimize: Option<bool>,
}

/// Response for model upload
#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub model_id: String,
    pub message: String,
}

/// Request body for running inference
#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub inputs: Vec<TensorData>,
}

/// Response for inference
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub outputs: Vec<TensorData>,
    pub inference_time_ms: f64,
}

/// Tensor data for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub name: String,
    pub data: Vec<f64>,
    pub dims: Vec<usize>,
    pub dtype: String,
}

impl TensorData {
    /// Convert to internal Tensor format
    pub fn to_tensor(&self) -> Result<Tensor, ApiError> {
        // Create ndarray from data and dims
        let array = match ndarray::Array::from_shape_vec(self.dims.clone(), self.data.clone()) {
            Ok(arr) => arr.into_dyn(),
            Err(e) => return Err(ApiError::BadRequest(format!("Invalid tensor shape: {:?}", e))),
        };

        // Convert to Tensor
        Tensor::try_from(array)
            .map_err(|e| ApiError::BadRequest(format!("Failed to create tensor: {:?}", e)))
    }

    /// Create from internal Tensor format
    pub fn from_tensor(name: String, tensor: &Tensor) -> Self {
        let dims = tensor.dims.inner().to_vec();

        // Extract data as f64 vector
        let data = match &tensor.data {
            my_onnx::tensor::data::TensorData::Float(_, v) => v.clone(),
            my_onnx::tensor::data::TensorData::SInt(_, v) => v.iter().map(|&x| x as f64).collect(),
            my_onnx::tensor::data::TensorData::UInt(_, v) => v.iter().map(|&x| x as f64).collect(),
        };

        let dtype = format!("{:?}", tensor.data.elem_type());

        Self {
            name,
            data,
            dims,
            dtype,
        }
    }
}

/// Model list item
#[derive(Debug, Serialize)]
pub struct ModelListItem {
    pub model_id: String,
    pub uploaded_at: String,
    pub status: String,
}

/// POST /models/upload - Upload and compile an ONNX model
pub async fn upload_model(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, ApiError> {
    let mut model_file: Option<Vec<u8>> = None;
    let mut target = Target::CPU;
    let mut optimize = true;

    // Parse multipart form data
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::BadRequest(format!("Invalid multipart: {:?}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::BadRequest(format!("Failed to read file: {:?}", e)))?;
                model_file = Some(data.to_vec());
            }
            "target" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| ApiError::BadRequest(format!("Failed to read target: {:?}", e)))?;
                target = match text.to_uppercase().as_str() {
                    "CPU" => Target::CPU,
                    "CUDA" => Target::CUDA,
                    _ => return Err(ApiError::BadRequest("Invalid target. Use CPU or CUDA".into())),
                };
            }
            "optimize" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| ApiError::BadRequest(format!("Failed to read optimize: {:?}", e)))?;
                optimize = text.parse().unwrap_or(true);
            }
            _ => {}
        }
    }

    let model_bytes = model_file.ok_or_else(|| ApiError::BadRequest("No model file provided".into()))?;

    // Save model file to upload directory
    let model_path = state.upload_dir.join(format!("model_{}.onnx", uuid::Uuid::new_v4()));
    std::fs::write(&model_path, &model_bytes)
        .map_err(|e| ApiError::InternalError(format!("Failed to save model: {:?}", e)))?;

    // Build options
    let options = Options::builder()
        .target(target)
        .enable_fuse_ops(optimize)
        .build();

    // Compile and cache the model
    let (model_id, _session) = state.cache.get_or_create(&model_path, options)?;

    Ok(Json(UploadResponse {
        model_id,
        message: "Model uploaded and compiled successfully".into(),
    }))
}

/// POST /models/:id/infer - Run inference on a model (JSON)
pub async fn run_inference(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, ApiError> {
    // Get cached session
    let session = state
        .cache
        .get(&model_id)
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", model_id)))?;

    // Convert input data to tensors
    let inputs: Vec<Tensor> = request
        .inputs
        .iter()
        .map(|td| td.to_tensor())
        .collect::<Result<Vec<_>, _>>()?;

    // Run inference with timing
    let start = std::time::Instant::now();
    let outputs = session.run(&inputs)?;
    let inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Convert outputs to JSON format
    let output_data: Vec<TensorData> = outputs
        .iter()
        .enumerate()
        .map(|(i, t)| TensorData::from_tensor(format!("output_{}", i), t))
        .collect();

    Ok(Json(InferenceResponse {
        outputs: output_data,
        inference_time_ms,
    }))
}

/// POST /models/:id/infer/proto - Run inference on a model (Protobuf)
/// Accepts: application/octet-stream (concatenated TensorProto bytes)
/// Returns: application/octet-stream (concatenated TensorProto bytes)
pub async fn run_inference_proto(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    body: Bytes,
) -> Result<impl IntoResponse, ApiError> {
    // Get cached session
    let session = state
        .cache
        .get(&model_id)
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", model_id)))?;

    // Decode input tensors from protobuf
    // Format: [4 bytes length][tensor proto bytes][4 bytes length][tensor proto bytes]...
    let mut inputs = Vec::new();
    let mut offset = 0;
    while offset < body.len() {
        if offset + 4 > body.len() {
            return Err(ApiError::BadRequest("Invalid protobuf format".into()));
        }

        let len = u32::from_le_bytes([body[offset], body[offset+1], body[offset+2], body[offset+3]]) as usize;
        offset += 4;

        if offset + len > body.len() {
            return Err(ApiError::BadRequest("Invalid protobuf format".into()));
        }

        let tensor_bytes = &body[offset..offset+len];
        let tensor = Tensor::from_proto_bytes(tensor_bytes)?;
        inputs.push(tensor);
        offset += len;
    }

    // Run inference
    let outputs = session.run(&inputs)?;

    // Encode output tensors to protobuf
    let mut response_bytes = Vec::new();
    for tensor in outputs.iter() {
        let proto_bytes = tensor.to_proto_bytes();
        let len = proto_bytes.len() as u32;
        response_bytes.extend_from_slice(&len.to_le_bytes());
        response_bytes.extend_from_slice(&proto_bytes);
    }

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        response_bytes,
    ))
}

/// GET /models - List all cached models
pub async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<Vec<ModelListItem>>, ApiError> {
    let models = state.cache.list_models();

    let items: Vec<ModelListItem> = models
        .into_iter()
        .map(|m| ModelListItem {
            model_id: m.model_id,
            uploaded_at: format!("{:?}", m.uploaded_at),
            status: "ready".into(),
        })
        .collect();

    Ok(Json(items))
}

/// GET / - Serve the main UI
pub async fn serve_ui() -> impl IntoResponse {
    let html = include_str!("../dist/index.html");
    (StatusCode::OK, [("Content-Type", "text/html")], html)
}
