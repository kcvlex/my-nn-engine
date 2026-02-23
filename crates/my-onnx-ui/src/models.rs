use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use my_onnx::options::Options;
use my_onnx::options::Target;
use my_onnx::session::Session;
use my_onnx::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    Mnist,
    ResNet,
    Yolo,
    Bert,
    Gpt2,
}

impl ModelId {
    pub fn model_path(&self) -> &'static str {
        match self {
            ModelId::Mnist => "mnist-12/mnist-12.onnx",
            ModelId::ResNet => "resnet18-v2-7/resnet18-v2-7.onnx",
            ModelId::Yolo => "yolov4/yolov4.onnx",
            ModelId::Bert => "bertsquad-12/bertsquad-12.onnx",
            ModelId::Gpt2 => "GPT2/model.onnx",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ModelId::Mnist => "MNIST",
            ModelId::ResNet => "ResNet",
            ModelId::Yolo => "YOLO",
            ModelId::Bert => "BERT",
            ModelId::Gpt2 => "GPT-2",
        }
    }
}

pub struct ModelRegistry {
    models: HashMap<(ModelId, Target), Arc<Session>>,
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

    pub fn get_or_load(
        &mut self,
        model_id: ModelId,
        target: Target,
        inputs: &[Tensor],
    ) -> Result<Arc<Session>, String> {
        // TODO: Type check.
        if let Some(session) = self.models.get(&(model_id, target)) {
            return Ok(session.clone());
        }

        let model_path = self.models_dir.join(model_id.model_path());
        log::info!(
            "Loading model: {} from {}",
            model_id.display_name(),
            model_path.display()
        );

        let options = Options::builder().target(target).build();

        let types = inputs.iter().map(|t| t.tensor_type()).collect::<Vec<_>>();
        let session = Session::new(&model_path, Some(types.as_slice()), &options)
            .map_err(|e| format!("Failed to load {}: {:?}", model_id.display_name(), e))?;

        let session = Arc::new(session);
        self.models.insert((model_id, target), session.clone());
        log::info!("Loaded {} successfully", model_id.display_name());
        Ok(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_model_path() {
        assert!(ModelId::Mnist.model_path().contains("mnist"));
        assert!(ModelId::ResNet.model_path().contains("resnet"));
        assert!(ModelId::Yolo.model_path().contains("yolo"));
        assert!(ModelId::Bert.model_path().contains("bert"));
        assert!(ModelId::Gpt2.model_path().to_lowercase().contains("gpt2"));
    }

    #[test]
    fn test_model_id_display_name() {
        assert_eq!(ModelId::Mnist.display_name(), "MNIST");
        assert_eq!(ModelId::ResNet.display_name(), "ResNet");
        assert_eq!(ModelId::Yolo.display_name(), "YOLO");
        assert_eq!(ModelId::Bert.display_name(), "BERT");
        assert_eq!(ModelId::Gpt2.display_name(), "GPT-2");
    }

    #[test]
    fn test_model_registry_new() {
        let registry = ModelRegistry::new("/tmp/models");
        assert_eq!(registry.models_dir, PathBuf::from("/tmp/models"));
        assert!(registry.models.is_empty());
    }

    #[test]
    fn test_model_registry_model_path() {
        let registry = ModelRegistry::new("/tmp/models");
        let path = registry.model_path(ModelId::Mnist);
        assert_eq!(
            path,
            PathBuf::from("/tmp/models").join(ModelId::Mnist.model_path())
        );
    }

    #[test]
    fn test_get_or_load_returns_error_for_missing_file() {
        let mut registry = ModelRegistry::new("/tmp/nonexistent_models_dir");
        let target = Target::CPU;
        let result = registry.get_or_load(ModelId::Mnist, target, &[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("MNIST"), "Error should mention model name");
        assert!(
            err.contains("Failed to load"),
            "Error should indicate a load failure"
        );
    }
}
