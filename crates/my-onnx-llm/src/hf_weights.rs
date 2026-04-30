use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use my_onnx::onnx::model::ExternalTensorRef;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use safetensors::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum HfWeightsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("weight not found: {0}")]
    NotFound(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightEntry {
    offset: u64,
    length: u64,
    shape: Vec<usize>,
}

pub struct HfWeights {
    bin_path: PathBuf,
    entries: HashMap<String, WeightEntry>,
}

fn bf16_bytes_to_f32_bytes(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for c in src.chunks_exact(2) {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        let f32_bits = (bits as u32) << 16;
        out.extend_from_slice(&f32_bits.to_le_bytes());
    }
    out
}

impl HfWeights {
    /// Convert a safetensors file (BF16/F32) into a flat F32 binary plus a JSON
    /// index. Idempotent: returns immediately if both outputs already exist.
    /// Weights are stored in their HF-natural layout (no transpose); ONNX
    /// MatMul callers should insert an explicit `Transpose` upstream and rely
    /// on `Canonicalization` to absorb it into the `Gemm.trans_b` flag.
    pub fn preprocess_safetensors(
        src: impl AsRef<Path>,
        bin_out: impl AsRef<Path>,
        index_out: impl AsRef<Path>,
    ) -> Result<(), HfWeightsError> {
        let bin_out = bin_out.as_ref();
        let index_out = index_out.as_ref();
        if bin_out.exists() && index_out.exists() {
            return Ok(());
        }

        let file = std::fs::File::open(src)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        let mut bin = std::io::BufWriter::new(std::fs::File::create(bin_out)?);
        let mut index: BTreeMap<String, WeightEntry> = BTreeMap::new();
        let mut offset = 0u64;

        for (name, view) in st.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let f32_bytes = match view.dtype() {
                Dtype::F32 => view.data().to_vec(),
                Dtype::BF16 => bf16_bytes_to_f32_bytes(view.data()),
                d => return Err(HfWeightsError::UnsupportedDtype(d)),
            };

            let len = f32_bytes.len() as u64;
            bin.write_all(&f32_bytes)?;
            index.insert(
                name.to_string(),
                WeightEntry {
                    offset,
                    length: len,
                    shape,
                },
            );
            offset += len;
        }

        bin.flush()?;
        drop(bin);

        let json = serde_json::to_string_pretty(&index)?;
        std::fs::write(index_out, json)?;
        Ok(())
    }

    pub fn from_index(
        bin_path: impl AsRef<Path>,
        index_path: impl AsRef<Path>,
    ) -> Result<Self, HfWeightsError> {
        let json = std::fs::read_to_string(index_path)?;
        let entries: HashMap<String, WeightEntry> = serde_json::from_str(&json)?;
        Ok(Self {
            bin_path: bin_path.as_ref().to_path_buf(),
            entries,
        })
    }

    pub fn external_ref(&self, name: &str) -> Result<ExternalTensorRef, HfWeightsError> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| HfWeightsError::NotFound(name.to_string()))?;
        Ok(ExternalTensorRef {
            path: self.bin_path.clone(),
            offset: e.offset,
            length: Some(e.length),
            elem_type: DataType::Float(FloatType::F32),
            dims: ResolvedTensorDims::new(&e.shape),
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }
}
