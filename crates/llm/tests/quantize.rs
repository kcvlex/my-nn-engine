use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use my_nn_engine_llm::quantize::quantize_safetensors_int8;
use my_nn_engine_llm::quantize::quantize_safetensors_int8_streaming;
use my_nn_engine_llm::quantize::QuantizeStats;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::View;
use tempfile::TempDir;

struct TestTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for TestTensor {
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

fn f32_tensor(shape: Vec<usize>, data: &[f32]) -> TestTensor {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    TestTensor {
        dtype: Dtype::F32,
        shape,
        data: bytes,
    }
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn read_bf16_array(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn read_i8_array(bytes: &[u8]) -> Vec<i8> {
    bytes.iter().map(|&b| b as i8).collect()
}

fn read_f32_array(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn write_shard(path: &Path, tensors: Vec<(String, TestTensor)>) {
    safetensors::serialize_to_file(tensors, None, path).unwrap();
}

fn assert_dequant_close(st: &SafeTensors, name: &str, orig: &[f32], shape: &[usize]) {
    let q = st.tensor(name).unwrap();
    assert_eq!(q.dtype(), Dtype::I8, "{name}: int8 dtype");
    assert_eq!(q.shape(), shape, "{name}: shape");

    let out_channels = shape[0];
    let inner: usize = shape[1..].iter().product();
    assert_eq!(orig.len(), out_channels * inner);

    let scale_view = st.tensor(&format!("{name}.scale")).unwrap();
    assert_eq!(scale_view.dtype(), Dtype::BF16, "{name}.scale: dtype");
    assert_eq!(scale_view.shape(), &[out_channels], "{name}.scale: shape");

    let int8 = read_i8_array(q.data());
    let scales = read_bf16_array(scale_view.data());
    for ch in 0..out_channels {
        let row = &orig[ch * inner..(ch + 1) * inner];
        let max_abs = row.iter().fold(0.0f32, |a, &b| a.max(b.abs())).max(1e-6);
        let tol = max_abs / 127.0;
        for j in 0..inner {
            let dq = int8[ch * inner + j] as f32 * scales[ch];
            let want = orig[ch * inner + j];
            assert!(
                (dq - want).abs() <= tol,
                "{name} ch{ch} j{j}: dq={dq} orig={want} tol={tol}",
            );
        }
    }
}

#[test]
fn quantize_int8_single_file_quantizes_targets_and_passes_others() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("in.safetensors");
    let dst = tmp.path().join("out.safetensors");

    // 3 channels x 4 cols. Picked so per-channel scales are easy to reason about:
    //   ch0: max_abs = 127  -> scale = 1.0
    //   ch1: max_abs = 2.0  -> scale = 2/127
    //   ch2: all zeros      -> scale = 1.0 (zero-row guard)
    let q_proj_data = vec![
        127.0, -127.0, 0.0, 64.0, //
        2.0, -1.0, 0.0, 0.5, //
        0.0, 0.0, 0.0, 0.0,
    ];
    let q_proj = f32_tensor(vec![3, 4], &q_proj_data);

    // input_layernorm.weight is not in the QUANT_PATTERNS list -> passthrough.
    let norm_data = vec![1.0, 2.0, 3.0, 4.0];
    let norm = f32_tensor(vec![4], &norm_data);

    write_shard(
        &src,
        vec![
            ("model.layers.0.self_attn.q_proj.weight".to_string(), q_proj),
            ("model.layers.0.input_layernorm.weight".to_string(), norm),
        ],
    );

    let stats = quantize_safetensors_int8(&src, &dst).unwrap();
    assert_eq!(
        stats,
        QuantizeStats {
            quantized: 1,
            passthrough: 1,
        },
    );

    let bytes = std::fs::read(&dst).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let names: BTreeSet<_> = st.names().into_iter().collect();
    assert_eq!(
        names,
        BTreeSet::from([
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight.scale",
            "model.layers.0.input_layernorm.weight",
        ]),
    );

    let q_name = "model.layers.0.self_attn.q_proj.weight";
    assert_dequant_close(&st, q_name, &q_proj_data, &[3, 4]);

    // Per-channel scale = max_abs(row) / 127, stored as BF16.
    let scales = read_bf16_array(st.tensor(&format!("{q_name}.scale")).unwrap().data());
    let expected_scales = [1.0, 2.0 / 127.0, 1.0];
    assert_eq!(scales.len(), expected_scales.len());
    for (ch, (got, want)) in scales.iter().zip(expected_scales).enumerate() {
        assert!((got - want).abs() < 1e-3, "ch{ch}: scale {got} != {want}");
    }

    // Passthrough stays F32 byte-for-byte (copy_view leaves F32 alone).
    let norm_view = st.tensor("model.layers.0.input_layernorm.weight").unwrap();
    assert_eq!(norm_view.dtype(), Dtype::F32);
    assert_eq!(norm_view.shape(), &[4]);
    assert_eq!(read_f32_array(norm_view.data()), norm_data);
}

#[test]
fn quantize_int8_to_dir_writes_per_shard_with_index() {
    let tmp = TempDir::new().unwrap();
    let src_dir = tmp.path().join("src");
    let dst_dir = tmp.path().join("dst");
    std::fs::create_dir_all(&src_dir).unwrap();

    let q_proj_data = vec![127.0, 0.0, -127.0, 64.0, 1.0, 2.0, -1.0, 0.5];
    let norm_data = vec![1.0, 2.0, 3.0, 4.0];
    write_shard(
        &src_dir.join("shard1.safetensors"),
        vec![
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                f32_tensor(vec![2, 4], &q_proj_data),
            ),
            (
                "model.layers.0.input_layernorm.weight".to_string(),
                f32_tensor(vec![4], &norm_data),
            ),
        ],
    );

    let lm_head_data = vec![1.0, -0.5, 0.25, 0.125, -1.0, 0.5];
    let embed_data = vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.0, 0.0, 0.125, -2.0, 1.0];
    write_shard(
        &src_dir.join("shard2.safetensors"),
        vec![
            (
                "lm_head.weight".to_string(),
                f32_tensor(vec![3, 2], &lm_head_data),
            ),
            (
                "model.embed_tokens.weight".to_string(),
                f32_tensor(vec![5, 2], &embed_data),
            ),
        ],
    );

    let index = serde_json::json!({
        "metadata": {"total_size": 0u64},
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "shard1.safetensors",
            "model.layers.0.input_layernorm.weight": "shard1.safetensors",
            "lm_head.weight": "shard2.safetensors",
            "model.embed_tokens.weight": "shard2.safetensors",
        }
    });
    std::fs::write(
        src_dir.join("model.safetensors.index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();

    let stats = quantize_safetensors_int8_streaming(&src_dir, &dst_dir).unwrap();
    assert_eq!(
        stats,
        QuantizeStats {
            quantized: 3,
            passthrough: 1,
        },
    );

    assert!(dst_dir.join("shard1.safetensors").exists());
    assert!(dst_dir.join("shard2.safetensors").exists());
    let index_path = dst_dir.join("model.safetensors.index.json");
    assert!(index_path.exists());

    let out_index: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&index_path).unwrap()).unwrap();
    let weight_map: BTreeMap<&str, &str> = out_index["weight_map"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str().unwrap()))
        .collect();
    assert_eq!(
        weight_map,
        BTreeMap::from([
            (
                "model.layers.0.self_attn.q_proj.weight",
                "shard1.safetensors"
            ),
            (
                "model.layers.0.self_attn.q_proj.weight.scale",
                "shard1.safetensors"
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "shard1.safetensors"
            ),
            ("lm_head.weight", "shard2.safetensors"),
            ("lm_head.weight.scale", "shard2.safetensors"),
            ("model.embed_tokens.weight", "shard2.safetensors"),
            ("model.embed_tokens.weight.scale", "shard2.safetensors"),
        ]),
    );

    let reported_size = out_index["metadata"]["total_size"].as_u64().unwrap();
    let actual_size: u64 = ["shard1.safetensors", "shard2.safetensors"]
        .iter()
        .map(|shard| {
            let bytes = std::fs::read(dst_dir.join(shard)).unwrap();
            SafeTensors::deserialize(&bytes)
                .unwrap()
                .tensors()
                .into_iter()
                .map(|(_, view)| view.data().len() as u64)
                .sum::<u64>()
        })
        .sum();
    assert_eq!(reported_size, actual_size);

    let bytes1 = std::fs::read(dst_dir.join("shard1.safetensors")).unwrap();
    let st1 = SafeTensors::deserialize(&bytes1).unwrap();
    assert_dequant_close(
        &st1,
        "model.layers.0.self_attn.q_proj.weight",
        &q_proj_data,
        &[2, 4],
    );
    let norm_view = st1.tensor("model.layers.0.input_layernorm.weight").unwrap();
    assert_eq!(norm_view.dtype(), Dtype::F32);
    assert_eq!(read_f32_array(norm_view.data()), norm_data);

    let bytes2 = std::fs::read(dst_dir.join("shard2.safetensors")).unwrap();
    let st2 = SafeTensors::deserialize(&bytes2).unwrap();
    assert_dequant_close(&st2, "lm_head.weight", &lm_head_data, &[3, 2]);
    assert_dequant_close(&st2, "model.embed_tokens.weight", &embed_data, &[5, 2]);
}
