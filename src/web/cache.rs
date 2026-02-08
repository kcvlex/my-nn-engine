use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use sha2::{Sha256, Digest};

use crate::options::Options;
use crate::session::{Session, SessionError};

/// A unique identifier for a cached model session
pub type ModelId = String;

/// Metadata about a cached model
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub model_id: ModelId,
    pub model_path: PathBuf,
    pub options: Options,
    pub uploaded_at: std::time::SystemTime,
}

/// Thread-safe cache for compiled ONNX model sessions
pub struct SessionCache {
    sessions: RwLock<HashMap<ModelId, Arc<Session>>>,
    metadata: RwLock<HashMap<ModelId, ModelMetadata>>,
    cache_dir: PathBuf,
}

impl SessionCache {
    /// Create a new session cache with the given cache directory
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir).ok();

        Self {
            sessions: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            cache_dir,
        }
    }

    /// Compute a unique hash for a model file and options
    fn compute_model_hash(model_path: &Path, options: &Options) -> Result<ModelId, SessionError> {
        let mut hasher = Sha256::new();

        // Hash the model file contents
        let model_bytes = std::fs::read(model_path)
            .map_err(|e| SessionError::OtherError(format!("Failed to read model: {:?}", e)))?;
        hasher.update(&model_bytes);

        // Hash the options (using Debug representation for simplicity)
        let options_str = format!("{:?}", options);
        hasher.update(options_str.as_bytes());

        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    /// Get or create a session for the given model and options
    pub fn get_or_create(
        &self,
        model_path: &Path,
        options: Options,
    ) -> Result<(ModelId, Arc<Session>), SessionError> {
        let model_id = Self::compute_model_hash(model_path, &options)?;

        // Check if we have a cached session
        {
            let sessions = self.sessions.read().unwrap();
            if let Some(session) = sessions.get(&model_id) {
                return Ok((model_id, Arc::clone(session)));
            }
        }

        // Session not cached, compile it
        log::info!("Compiling new session for model_id: {}", model_id);
        let session = Session::new(model_path, None, &options)?;
        let session = Arc::new(session);

        // Store in cache
        {
            let mut sessions = self.sessions.write().unwrap();
            sessions.insert(model_id.clone(), Arc::clone(&session));
        }

        // Store metadata
        {
            let mut metadata = self.metadata.write().unwrap();
            metadata.insert(
                model_id.clone(),
                ModelMetadata {
                    model_id: model_id.clone(),
                    model_path: model_path.to_path_buf(),
                    options,
                    uploaded_at: std::time::SystemTime::now(),
                },
            );
        }

        Ok((model_id, session))
    }

    /// Get an existing session by model ID
    pub fn get(&self, model_id: &str) -> Option<Arc<Session>> {
        let sessions = self.sessions.read().unwrap();
        sessions.get(model_id).map(Arc::clone)
    }

    /// Get metadata for a model
    pub fn get_metadata(&self, model_id: &str) -> Option<ModelMetadata> {
        let metadata = self.metadata.read().unwrap();
        metadata.get(model_id).cloned()
    }

    /// List all cached model IDs
    pub fn list_models(&self) -> Vec<ModelMetadata> {
        let metadata = self.metadata.read().unwrap();
        metadata.values().cloned().collect()
    }

    /// Remove a model from the cache
    pub fn remove(&self, model_id: &str) -> bool {
        let mut sessions = self.sessions.write().unwrap();
        let mut metadata = self.metadata.write().unwrap();

        sessions.remove(model_id).is_some() | metadata.remove(model_id).is_some()
    }

    /// Clear all cached sessions
    pub fn clear(&self) {
        let mut sessions = self.sessions.write().unwrap();
        let mut metadata = self.metadata.write().unwrap();

        sessions.clear();
        metadata.clear();
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}
