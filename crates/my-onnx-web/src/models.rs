use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use my_onnx::options::{Options, Target};
use my_onnx::session::Session;

/// Supported pre-defined models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    Mnist,
    ResNet,
    Yolo,
    Bert,
    Gpt2,
}

impl ModelId {
    /// Get the filename for this model
    pub fn filename(&self) -> &'static str {
        match self {
            ModelId::Mnist => "mnist.onnx",
            ModelId::ResNet => "resnet.onnx",
            ModelId::Yolo => "yolo.onnx",
            ModelId::Bert => "bert.onnx",
            ModelId::Gpt2 => "gpt2.onnx",
        }
    }

    /// Get a human-readable name for this model
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelId::Mnist => "MNIST",
            ModelId::ResNet => "ResNet",
            ModelId::Yolo => "YOLO",
            ModelId::Bert => "BERT",
            ModelId::Gpt2 => "GPT-2",
        }
    }

    /// All supported model IDs
    pub fn all() -> &'static [ModelId] {
        &[
            ModelId::Mnist,
            ModelId::ResNet,
            ModelId::Yolo,
            ModelId::Bert,
            ModelId::Gpt2,
        ]
    }
}

/// Registry of pre-loaded models
pub struct ModelRegistry {
    models: HashMap<(ModelId, Target), Arc<Session>>,
    models_dir: PathBuf,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        Self {
            models: HashMap::new(),
            models_dir: models_dir.as_ref().to_path_buf(),
        }
    }

    /// Load all models for a specific target
    pub fn load_models(&mut self, target: Target) -> Result<(), String> {
        log::info!("Loading models for target: {:?}", target);

        for model_id in ModelId::all() {
            let model_path = self.models_dir.join(model_id.filename());

            if !model_path.exists() {
                log::warn!(
                    "Model file not found: {} (skipping)",
                    model_path.display()
                );
                continue;
            }

            log::info!("Loading model: {}", model_id.display_name());

            let options = Options::builder()
                .target(target)
                .enable_fuse_ops(true)
                .build();

            match Session::from_path(&model_path, options) {
                Ok(session) => {
                    self.models.insert((*model_id, target), Arc::new(session));
                    log::info!("✓ Loaded {} successfully", model_id.display_name());
                }
                Err(e) => {
                    log::error!(
                        "Failed to load {}: {:?}",
                        model_id.display_name(),
                        e
                    );
                    return Err(format!(
                        "Failed to load {}: {:?}",
                        model_id.display_name(),
                        e
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get a model session by ID and target
    pub fn get(&self, model_id: ModelId, target: Target) -> Option<Arc<Session>> {
        self.models.get(&(model_id, target)).cloned()
    }

    /// Check if a model is loaded
    pub fn is_loaded(&self, model_id: ModelId, target: Target) -> bool {
        self.models.contains_key(&(model_id, target))
    }

    /// Get the number of loaded models
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}
