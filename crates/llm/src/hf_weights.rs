use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use itertools::Itertools;
use my_nn_engine::graph::ExternalTensorRef;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::SIntType;
use safetensors::Dtype;
use safetensors::SafeTensors;

#[derive(Debug, thiserror::Error)]
pub enum HfWeightsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("weight not found: {0}")]
    NotFound(String),
    #[error("malformed safetensors index: {0}")]
    MalformedIndex(&'static str),
    #[error(
        "no safetensors found in {}; only HuggingFace .safetensors layouts are supported. \
         Convert via `transformers.PreTrainedModel.save_pretrained(safe_serialization=True)`.",
        .0.display()
    )]
    MissingSafetensors(PathBuf),
}

#[derive(Debug, Clone)]
struct WeightEntry {
    path: PathBuf,
    offset: u64,
    length: u64,
    shape: Vec<usize>,
    elem_type: DataType,
}

#[derive(Default)]
pub struct HfWeights {
    entries: HashMap<String, WeightEntry>,
}

fn safetensors_dtype(d: Dtype) -> Result<DataType, HfWeightsError> {
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

pub const QUANTIZED_SAFETENSORS: &str = "model.int8.safetensors";
pub const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
pub const SINGLE_SAFETENSORS: &str = "model.safetensors";

impl HfWeights {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, HfWeightsError> {
        let dir = model_dir.as_ref();
        let int8 = dir.join(QUANTIZED_SAFETENSORS);
        if int8.exists() {
            return Self::from_safetensors(int8);
        }
        let index = dir.join(SAFETENSORS_INDEX);
        if index.exists() {
            return Self::from_index(index);
        }
        let single = dir.join(SINGLE_SAFETENSORS);
        if single.exists() {
            return Self::from_safetensors(single);
        }
        Err(HfWeightsError::MissingSafetensors(dir.to_path_buf()))
    }

    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self, HfWeightsError> {
        let mut res = Self::default();
        res.add_shard(path.as_ref())?;
        Ok(res)
    }

    pub fn from_index(index_path: impl AsRef<Path>) -> Result<Self, HfWeightsError> {
        let index_path = index_path.as_ref();
        let dir = index_path
            .parent()
            .ok_or(HfWeightsError::MalformedIndex("index has no parent dir"))?;
        let json: serde_json::Value = serde_json::from_reader(std::fs::File::open(index_path)?)?;
        let weight_map = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or(HfWeightsError::MalformedIndex("missing weight_map"))?;
        let mut res = Self::default();
        for shard in weight_map
            .values()
            .filter_map(|v| v.as_str())
            .unique()
            .map(|s| dir.join(s))
        {
            res.add_shard(&shard)?;
        }
        Ok(res)
    }

    fn add_shard(&mut self, path: &Path) -> Result<(), HfWeightsError> {
        let path = path.to_path_buf();
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let (header_bytes, metadata) = SafeTensors::read_metadata(&mmap)?;
        let data_start = (8 + header_bytes) as u64;
        for (name, info) in metadata.tensors() {
            let (start, end) = info.data_offsets;
            self.entries.insert(
                name.to_string(),
                WeightEntry {
                    path: path.clone(),
                    offset: data_start + start as u64,
                    length: (end - start) as u64,
                    shape: info.shape.to_vec(),
                    elem_type: safetensors_dtype(info.dtype)?,
                },
            );
        }
        Ok(())
    }

    pub fn external_ref(&self, name: &str) -> Result<ExternalTensorRef, HfWeightsError> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| HfWeightsError::NotFound(name.to_string()))?;
        Ok(ExternalTensorRef {
            path: e.path.clone(),
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
