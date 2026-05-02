use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::ResolvedTensorType;
use my_nn_engine::tensor::types::SIntType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    Mnist,
    ResNet,
    ResNet152,
    MobileNetV2,
    EfficientNetLite4,
    Yolo,
    Bert,
    Gpt2,
}

fn ty_f32(dims: &[usize]) -> ResolvedTensorType {
    ResolvedTensorType::new(
        DataType::Float(FloatType::F32),
        ResolvedTensorDims::new(dims),
    )
}

fn ty_i64(dims: &[usize]) -> ResolvedTensorType {
    ResolvedTensorType::new(
        DataType::SInt(SIntType::I64),
        ResolvedTensorDims::new(dims),
    )
}

impl ModelId {
    pub fn model_path(&self) -> &'static str {
        match self {
            ModelId::Mnist => "mnist-12/mnist-12.onnx",
            ModelId::ResNet => "resnet18-v2-7/resnet18-v2-7.onnx",
            ModelId::ResNet152 => "resnet152-v2-7/resnet152-v2-7.onnx",
            ModelId::MobileNetV2 => "mobilenetv2-12/mobilenetv2-12.onnx",
            ModelId::EfficientNetLite4 => "efficientnet-lite4-11/efficientnet-lite4.onnx",
            ModelId::Yolo => "yolov4/yolov4.onnx",
            ModelId::Bert => "bertsquad-12/bertsquad-12.onnx",
            ModelId::Gpt2 => "GPT2/model.onnx",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ModelId::Mnist => "MNIST",
            ModelId::ResNet => "ResNet18",
            ModelId::ResNet152 => "ResNet152",
            ModelId::MobileNetV2 => "MobileNetV2",
            ModelId::EfficientNetLite4 => "EfficientNet-Lite4",
            ModelId::Yolo => "YOLO",
            ModelId::Bert => "BERT",
            ModelId::Gpt2 => "GPT-2",
        }
    }

    /// Input tensor types in graph-input order. Used by ModelRegistry::warm_up
    /// so the session can be compiled before the first inference call.
    pub fn input_specs(&self) -> Vec<ResolvedTensorType> {
        match self {
            ModelId::Mnist => vec![ty_f32(&[1, 1, 28, 28])],
            ModelId::ResNet | ModelId::ResNet152 | ModelId::MobileNetV2 => {
                vec![ty_f32(&[1, 3, 224, 224])]
            }
            ModelId::EfficientNetLite4 => vec![ty_f32(&[1, 224, 224, 3])],
            ModelId::Yolo => vec![ty_f32(&[1, 416, 416, 3])],
            ModelId::Bert => vec![
                ty_i64(&[1]),
                ty_i64(&[1, 256]),
                ty_i64(&[1, 256]),
                ty_i64(&[1, 256]),
            ],
            ModelId::Gpt2 => vec![ty_i64(&[1, 1, 8])],
        }
    }
}

pub struct ModelRegistry {
    models: HashMap<(ModelId, Target), Arc<Mutex<Session>>>,
    models_dir: PathBuf,
}

impl ModelRegistry {
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        Self {
            models: HashMap::new(),
            models_dir: models_dir.as_ref().to_path_buf(),
        }
    }

    pub fn model_path(&self, model_id: ModelId) -> PathBuf {
        self.models_dir.join(model_id.model_path())
    }

    pub fn is_loaded(&self, model_id: ModelId, target: Target) -> bool {
        self.models.contains_key(&(model_id, target))
    }

    /// Build and cache a session for `(model_id, target)` if not already cached.
    pub fn warm_up(&mut self, model_id: ModelId, target: Target) -> Result<(), String> {
        if self.is_loaded(model_id, target) {
            return Ok(());
        }

        let model_path = self.models_dir.join(model_id.model_path());
        log::info!(
            "Loading model: {} from {}",
            model_id.display_name(),
            model_path.display()
        );

        let options = Options::builder().target(target).build();
        let input_types = model_id.input_specs();
        let session = Session::new(
            &model_path,
            Some(&input_types),
            &options,
            &SessionConfig::default(),
        )
        .map_err(|e| format!("Failed to load {}: {:?}", model_id.display_name(), e))?;

        self.models
            .insert((model_id, target), Arc::new(Mutex::new(session)));
        log::info!("Loaded {} successfully", model_id.display_name());
        Ok(())
    }

    pub fn get_or_load(
        &mut self,
        model_id: ModelId,
        target: Target,
    ) -> Result<Arc<Mutex<Session>>, String> {
        self.warm_up(model_id, target)?;
        Ok(self
            .models
            .get(&(model_id, target))
            .expect("warm_up just inserted this entry")
            .clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_new() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::new(tmp.path());
        assert_eq!(registry.models_dir, tmp.path());
        assert!(registry.models.is_empty());
    }

    #[test]
    fn test_model_registry_model_path() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::new(tmp.path());
        let path = registry.model_path(ModelId::Mnist);
        assert_eq!(path, tmp.path().join(ModelId::Mnist.model_path()));
    }

    #[test]
    fn test_get_or_load_returns_error_for_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let mut registry = ModelRegistry::new(tmp.path());
        let target = Target::CPU;
        let result = registry.get_or_load(ModelId::Mnist, target);
        assert!(result.is_err());
    }
}
