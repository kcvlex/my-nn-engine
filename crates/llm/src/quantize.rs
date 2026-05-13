use std::borrow::Cow;
use std::path::Path;

use safetensors::tensor::TensorView;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::View;

#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported source dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("expected >=2D weight, got shape {0:?}")]
    BadShape(Vec<usize>),
    #[error("malformed safetensors.index.json")]
    MalformedIndex,
}

const QUANT_PATTERNS: &[&str] = &[
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
    "lm_head.weight",
    "embed_tokens.weight",
];

fn should_quantize(name: &str) -> bool {
    QUANT_PATTERNS.iter().any(|p| name.ends_with(p))
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) & 0x1;
    let exp = ((bits as u32) >> 10) & 0x1F;
    let mantissa = (bits as u32) & 0x3FF;
    let f32_bits: u32 = if exp == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            let mut e: u32 = 1;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            let m = m & 0x3FF;
            (sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (mantissa << 13)
    } else {
        let new_exp = exp + (127 - 15);
        (sign << 31) | (new_exp << 23) | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_bf16_bits_rne(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let bits = x.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn read_as_f32(view: &TensorView<'_>) -> Result<Vec<f32>, QuantizeError> {
    let bytes = view.data();
    let v = match view.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        d => return Err(QuantizeError::UnsupportedDtype(d)),
    };
    Ok(v)
}

fn quantize_per_channel(
    view: &TensorView<'_>,
) -> Result<(OwnedTensor, OwnedTensor), QuantizeError> {
    let shape = view.shape().to_vec();
    if shape.len() < 2 {
        return Err(QuantizeError::BadShape(shape));
    }
    let out_channels = shape[0];
    let inner: usize = shape[1..].iter().product();
    let f32_data = read_as_f32(view)?;

    let mut scale = vec![0.0f32; out_channels];
    for ch in 0..out_channels {
        let row = &f32_data[ch * inner..(ch + 1) * inner];
        let max_abs = row.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        scale[ch] = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    }

    let mut int8 = vec![0i8; f32_data.len()];
    for ch in 0..out_channels {
        let s = scale[ch];
        for j in 0..inner {
            let q = (f32_data[ch * inner + j] / s).round().clamp(-128.0, 127.0);
            int8[ch * inner + j] = q as i8;
        }
    }

    let int8_bytes: Vec<u8> = int8.iter().map(|&x| x as u8).collect();
    let scale_bytes: Vec<u8> = scale
        .iter()
        .flat_map(|&s| f32_to_bf16_bits_rne(s).to_le_bytes())
        .collect();

    Ok((
        OwnedTensor {
            dtype: Dtype::I8,
            shape,
            data: int8_bytes,
        },
        OwnedTensor {
            dtype: Dtype::BF16,
            shape: vec![out_channels],
            data: scale_bytes,
        },
    ))
}

fn copy_view(view: &TensorView<'_>) -> Result<OwnedTensor, QuantizeError> {
    if view.dtype() == Dtype::F16 {
        let f32_data = read_as_f32(view)?;
        let bf16_bytes: Vec<u8> = f32_data
            .iter()
            .flat_map(|&f| f32_to_bf16_bits_rne(f).to_le_bytes())
            .collect();
        return Ok(OwnedTensor {
            dtype: Dtype::BF16,
            shape: view.shape().to_vec(),
            data: bf16_bytes,
        });
    }
    Ok(OwnedTensor {
        dtype: view.dtype(),
        shape: view.shape().to_vec(),
        data: view.data().to_vec(),
    })
}

fn cast_to_bf16(view: &TensorView<'_>) -> Result<OwnedTensor, QuantizeError> {
    let f32_data = read_as_f32(view)?;
    let bytes: Vec<u8> = f32_data
        .iter()
        .flat_map(|&x| f32_to_bf16_bits_rne(x).to_le_bytes())
        .collect();
    Ok(OwnedTensor {
        dtype: Dtype::BF16,
        shape: view.shape().to_vec(),
        data: bytes,
    })
}

fn list_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>, QuantizeError> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let json: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index)?)?;
        let map = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or(QuantizeError::MalformedIndex)?;
        let shards: std::collections::BTreeSet<&str> =
            map.values().filter_map(|v| v.as_str()).collect();
        Ok(shards.into_iter().map(|s| dir.join(s)).collect())
    } else {
        Ok(vec![dir.join("model.safetensors")])
    }
}

pub fn cast_safetensors_bf16_dir(
    src_dir: impl AsRef<Path>,
    dst: impl AsRef<Path>,
) -> Result<(), QuantizeError> {
    let dir = src_dir.as_ref();
    let files = list_shards(dir)?;

    let mut output: Vec<(String, OwnedTensor)> = Vec::new();
    for src in &files {
        let bytes = std::fs::read(src)?;
        let st = SafeTensors::deserialize(&bytes)?;
        for (name, view) in st.tensors() {
            let converted = match view.dtype() {
                Dtype::F32 | Dtype::BF16 => cast_to_bf16(&view)?,
                _ => copy_view(&view)?,
            };
            output.push((name, converted));
        }
    }
    safetensors::serialize_to_file(output, None, dst.as_ref())?;
    Ok(())
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QuantizeStats {
    pub quantized: usize,
    pub passthrough: usize,
}

pub fn quantize_safetensors_int8(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
) -> Result<QuantizeStats, QuantizeError> {
    quantize_safetensors_int8_files(&[src.as_ref().to_path_buf()], dst)
}

pub fn quantize_safetensors_int8_dir(
    src_dir: impl AsRef<Path>,
    dst: impl AsRef<Path>,
) -> Result<QuantizeStats, QuantizeError> {
    let files = list_shards(src_dir.as_ref())?;
    quantize_safetensors_int8_files(&files, dst)
}

// Streaming quantization.
pub fn quantize_safetensors_int8_to_dir(
    src_dir: impl AsRef<Path>,
    dst_dir: impl AsRef<Path>,
) -> Result<QuantizeStats, QuantizeError> {
    let src_dir = src_dir.as_ref();
    let dst_dir = dst_dir.as_ref();
    std::fs::create_dir_all(dst_dir)?;

    let index_path = src_dir.join("model.safetensors.index.json");
    let shard_names: Vec<String> = list_shards(src_dir)?
        .into_iter()
        .map(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(String::from)
                .ok_or(QuantizeError::MalformedIndex)
        })
        .collect::<Result<_, _>>()?;

    let mut stats = QuantizeStats::default();
    let mut weight_map = serde_json::Map::new();
    let mut total_size: u64 = 0;

    for shard_name in &shard_names {
        let src = src_dir.join(shard_name);
        let bytes = std::fs::read(&src)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut output: Vec<(String, OwnedTensor)> = Vec::new();
        for (name, view) in st.tensors() {
            if should_quantize(&name) {
                let (q, scale) = quantize_per_channel(&view)?;
                let scale_name = format!("{name}.scale");
                total_size += q.data.len() as u64 + scale.data.len() as u64;
                weight_map.insert(name.clone(), serde_json::Value::String(shard_name.clone()));
                weight_map.insert(
                    scale_name.clone(),
                    serde_json::Value::String(shard_name.clone()),
                );
                output.push((name.clone(), q));
                output.push((scale_name, scale));
                stats.quantized += 1;
            } else {
                let owned = copy_view(&view)?;
                total_size += owned.data.len() as u64;
                weight_map.insert(name.clone(), serde_json::Value::String(shard_name.clone()));
                output.push((name, owned));
                stats.passthrough += 1;
            }
        }
        let dst = dst_dir.join(shard_name);
        safetensors::serialize_to_file(output, None, &dst)?;
    }

    if shard_names.len() > 1 || index_path.exists() {
        let index_out = serde_json::json!({
            "metadata": { "total_size": total_size },
            "weight_map": serde_json::Value::Object(weight_map),
        });
        std::fs::write(
            dst_dir.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index_out)?,
        )?;
    }

    Ok(stats)
}

fn quantize_safetensors_int8_files(
    src_files: &[std::path::PathBuf],
    dst: impl AsRef<Path>,
) -> Result<QuantizeStats, QuantizeError> {
    let mut output: Vec<(String, OwnedTensor)> = Vec::new();
    let mut stats = QuantizeStats::default();
    for src in src_files {
        let bytes = std::fs::read(src)?;
        let st = SafeTensors::deserialize(&bytes)?;
        for (name, view) in st.tensors() {
            if should_quantize(&name) {
                let (q, scale) = quantize_per_channel(&view)?;
                output.push((name.clone(), q));
                output.push((format!("{name}.scale"), scale));
                stats.quantized += 1;
            } else {
                output.push((name.clone(), copy_view(&view)?));
                stats.passthrough += 1;
            }
        }
    }
    safetensors::serialize_to_file(output, None, dst.as_ref())?;
    Ok(stats)
}
