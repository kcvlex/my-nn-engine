use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::tensor::Tensor;

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

    pub fn get_or_load(
        &mut self,
        model_id: ModelId,
        target: Target,
        inputs: &[Tensor],
    ) -> Result<Arc<Mutex<Session>>, String> {
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
        let session = Session::new(
            &model_path,
            Some(types.as_slice()),
            &options,
            &SessionConfig::default(),
        )
        .map_err(|e| format!("Failed to load {}: {:?}", model_id.display_name(), e))?;

        let session = Arc::new(Mutex::new(session));
        self.models.insert((model_id, target), session.clone());
        log::info!("Loaded {} successfully", model_id.display_name());
        Ok(session)
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
        let result = registry.get_or_load(ModelId::Mnist, target, &[]);
        assert!(result.is_err());
    }
}
