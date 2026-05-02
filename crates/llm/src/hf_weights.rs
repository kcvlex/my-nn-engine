use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use my_nn_engine::graph::ExternalTensorRef;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use safetensors::Dtype;
use safetensors::SafeTensors;

#[derive(Debug, thiserror::Error)]
pub enum HfWeightsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("weight not found: {0}")]
    NotFound(String),
}

#[derive(Debug, Clone)]
struct WeightEntry {
    offset: u64,
    length: u64,
    shape: Vec<usize>,
    elem_type: DataType,
}

pub struct HfWeights {
    safetensors_path: PathBuf,
    entries: HashMap<String, WeightEntry>,
}

fn safetensors_dtype(d: Dtype) -> Result<DataType, HfWeightsError> {
    use my_nn_engine::tensor::types::SIntType;
    match d {
        Dtype::F32 => Ok(FloatType::F32.into()),
        Dtype::BF16 => Ok(FloatType::BF16.into()),
        Dtype::I8 => Ok(SIntType::I8.into()),
        d => Err(HfWeightsError::UnsupportedDtype(d)),
    }
}

#[derive(Debug, Clone)]
pub struct WeightRef {
    pub weight: ExternalTensorRef,
    pub scale: Option<ExternalTensorRef>,
}

impl HfWeights {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, HfWeightsError> {
        Self::from_safetensors(model_dir.as_ref().join("model.safetensors"))
    }

    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self, HfWeightsError> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let (header_bytes, metadata) = SafeTensors::read_metadata(&mmap)?;
        let data_start = (8 + header_bytes) as u64;
        let mut entries = HashMap::new();
        for (name, info) in metadata.tensors() {
            let (start, end) = info.data_offsets;
            entries.insert(
                name.to_string(),
                WeightEntry {
                    offset: data_start + start as u64,
                    length: (end - start) as u64,
                    shape: info.shape.to_vec(),
                    elem_type: safetensors_dtype(info.dtype)?,
                },
            );
        }
        Ok(Self {
            safetensors_path: path,
            entries,
        })
    }

    pub fn external_ref(&self, name: &str) -> Result<ExternalTensorRef, HfWeightsError> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| HfWeightsError::NotFound(name.to_string()))?;
        Ok(ExternalTensorRef {
            path: self.safetensors_path.clone(),
            offset: e.offset,
            length: Some(e.length),
            elem_type: e.elem_type,
            dims: ResolvedTensorDims::new(&e.shape),
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    pub fn weight_ref(&self, name: &str) -> Result<WeightRef, HfWeightsError> {
        let weight = self.external_ref(name)?;
        let scale = self.external_ref(&format!("{name}.scale")).ok();
        Ok(WeightRef { weight, scale })
    }
}
