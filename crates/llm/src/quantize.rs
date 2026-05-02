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
    #[error("unsupported source dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("expected >=2D weight, got shape {0:?}")]
    BadShape(Vec<usize>),
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

fn copy_view(view: &TensorView<'_>) -> OwnedTensor {
    OwnedTensor {
        dtype: view.dtype(),
        shape: view.shape().to_vec(),
        data: view.data().to_vec(),
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct QuantizeStats {
    pub quantized: usize,
    pub passthrough: usize,
}

pub fn quantize_safetensors_int8(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
) -> Result<QuantizeStats, QuantizeError> {
    let bytes = std::fs::read(src.as_ref())?;
    let st = SafeTensors::deserialize(&bytes)?;

    let mut output: Vec<(String, OwnedTensor)> = Vec::new();
    let mut stats = QuantizeStats::default();
    for (name, view) in st.tensors() {
        if should_quantize(&name) {
            let (q, scale) = quantize_per_channel(&view)?;
            output.push((name.clone(), q));
            output.push((format!("{name}.scale"), scale));
            stats.quantized += 1;
        } else {
            output.push((name.clone(), copy_view(&view)));
            stats.passthrough += 1;
        }
    }

    safetensors::serialize_to_file(output, None, dst.as_ref())?;
    Ok(stats)
}
