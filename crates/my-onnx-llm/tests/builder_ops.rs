use std::path::PathBuf;

use my_onnx::onnx::load::LoadProto;
use my_onnx::onnx::model::ValueId;
use my_onnx::options::Options;
use my_onnx::options::Target;
use my_onnx::session::Session;
use my_onnx::session::SessionConfig;
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::data::TensorData;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::SIntType;
use my_onnx::tensor::Tensor;
use my_onnx_llm::builder::Builder;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(dir: &str) -> PathBuf {
    workspace_root().join("models/test/single_op").join(dir)
}

fn load_pb(path: PathBuf) -> Option<Tensor> {
    Tensor::load_from_path(path).ok()
}

fn run_builder(graph: my_onnx::onnx::model::Graph, inputs: &[Tensor]) -> Tensor {
    run_builder_with_target(graph, inputs, Target::CPU)
}

fn run_builder_with_target(
    graph: my_onnx::onnx::model::Graph,
    inputs: &[Tensor],
    target: Target,
) -> Tensor {
    let opts = Options::builder().target(target).build();
    let mut session = Session::from_graph(graph, &opts, &SessionConfig::default()).unwrap();
    session.run(inputs).unwrap().into_iter().next().unwrap()
}

fn make_f32(dims: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::new(dims),
        TensorData::Float(FloatType::F32, data),
    )
    .unwrap()
}

fn input_for(b: &mut Builder, name: &str, t: &Tensor) -> ValueId {
    let ty = t.tensor_type();
    b.input(
        name,
        ty.elem_type,
        &ty.dims.iter().copied().collect::<Vec<_>>(),
    )
}

fn i64_init(b: &mut Builder, name: &str, values: Vec<i64>) -> ValueId {
    let len = values.len();
    let t = Tensor::new(
        ResolvedTensorDims::new(&[len]),
        TensorData::SInt(SIntType::I64, values),
    )
    .unwrap();
    b.initializer(name, t)
}

fn verify_op<F>(name: &str, f: F)
where
    F: FnOnce(&mut Builder, &[ValueId]) -> ValueId,
{
    let dir = fixture(name);
    let inputs = (0..)
        .map(|i| load_pb(dir.join(format!("input_{}.pb", i))))
        .take_while(|t| t.is_some())
        .map(|t| t.unwrap())
        .collect::<Vec<_>>();
    let expected = load_pb(dir.join("output_0.pb")).unwrap();

    let mut builder = Builder::new(&format!("test_{}", name));
    let input_ids = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| input_for(&mut builder, &format!("input_{}", i), t))
        .collect::<Vec<_>>();
    let out = f(&mut builder, &input_ids[..]);
    builder.output(out);

    let got = run_builder(builder.graph, &inputs);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}

#[test]
fn gather_default_axis() {
    verify_op("gather_default_axis", |b, input_ids| {
        b.gather("g", input_ids[0], input_ids[1], 0)
    });
}

#[test]
fn rms_norm_basic() {
    let x = make_f32(&[1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    let scale = make_f32(&[4], vec![0.5, 0.5, 0.5, 0.5]);
    let eps = 1e-5_f64;
    let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let expected = make_f32(
        &[1, 4],
        vec![
            0.5 * 1.0 * inv_rms,
            0.5 * 2.0 * inv_rms,
            0.5 * 3.0 * inv_rms,
            0.5 * 4.0 * inv_rms,
        ],
    );

    let mut builder = Builder::new("test_rms_norm");
    let x_in = builder.input("x", x.tensor_type().elem_type, &[1, 4]);
    let scale_in = builder.initializer("scale", scale);
    let out = builder.rms_norm("rms", x_in, scale_in, -1, eps);
    builder.output(out);

    let got = run_builder(builder.graph, &[x]);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}

#[test]
fn add() {
    verify_op("add", |b, input_ids| {
        b.add("add", input_ids[0], input_ids[1])
    });
}

#[test]
fn mul() {
    let a = make_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = make_f32(&[2, 3], vec![2.0, 2.0, 2.0, 0.5, 0.5, 0.5]);
    let expected = make_f32(&[2, 3], vec![2.0, 4.0, 6.0, 2.0, 2.5, 3.0]);

    let mut builder = Builder::new("test_mul");
    let lhs = input_for(&mut builder, "a", &a);
    let rhs = input_for(&mut builder, "b", &b);
    let out = builder.mul("mul", lhs, rhs);
    builder.output(out);

    let got = run_builder(builder.graph, &[a, b]);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}

#[test]
fn neg() {
    verify_op("neg", |b, input_ids| b.neg("neg", input_ids[0]));
}

#[test]
fn sigmoid() {
    verify_op("sigmoid", |b, input_ids| b.sigmoid("sigmoid", input_ids[0]));
}

#[test]
fn reshape() {
    verify_op("reshape", |b, input_ids| {
        let shape = i64_init(b, "shape", vec![3, 1, 1, 2, 4]);
        b.reshape("rs", input_ids[0], shape)
    });
}

#[test]
fn transpose() {
    verify_op("transpose", |b, input_ids| {
        b.transpose("tr", input_ids[0], vec![2, 3, 1, 0])
    });
}

#[test]
fn concat() {
    verify_op("concat_axis_2", |b, input_ids| {
        b.concat("cat", input_ids.to_vec(), 2)
    });
}

#[test]
fn slice() {
    verify_op("slice", |b, input_ids| {
        let starts = i64_init(b, "starts", vec![1, 0]);
        let ends = i64_init(b, "ends", vec![3, 4]);
        let axes = i64_init(b, "axes", vec![1, 2]);
        b.slice("sl", input_ids[0], starts, ends, Some(axes), None)
    });
}

#[test]
fn expand() {
    verify_op("expand", |b, input_ids| {
        let shape = i64_init(b, "shape", vec![3, 4]);
        b.expand("ex", input_ids[0], shape)
    });
}

#[test]
fn attention_no_causal() {
    verify_op("attention_no_causal", |b, input_ids| {
        let scale = 1.0_f32 / (8.0_f32).sqrt();
        b.attention(
            "attn",
            input_ids[0],
            input_ids[1],
            input_ids[2],
            None,
            None,
            false,
            scale,
        )
    });
}

#[cfg(feature = "cuda")]
#[test]
fn kv_cache_update() {
    let dir = fixture("kv_cache_update");
    let inputs: Vec<Tensor> = (0..3)
        .map(|i| load_pb(dir.join(format!("input_{}.pb", i))).unwrap())
        .collect();
    let expected = load_pb(dir.join("output_0.pb")).unwrap();

    let mut builder = Builder::new("test_kv_cache_update");
    let cache_in = input_for(&mut builder, "cache", &inputs[0]);
    let src_in = input_for(&mut builder, "src", &inputs[1]);
    let offset_in = input_for(&mut builder, "offset", &inputs[2]);
    let updated = builder.kv_cache_update("kvu", cache_in, src_in, offset_in);
    let out = builder.sigmoid("sig", updated);
    builder.output(out);

    let got = run_builder_with_target(builder.graph, &inputs, Target::CUDA);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}

#[test]
fn matmul_basic() {
    // A: [4, 3] @ B: [3, 2] -> C: [4, 2]
    // A = [[1..3], [4..6], [7..9], [10..12]]
    // B = [[1, 2], [3, 4], [5, 6]]
    // C = [[22, 28], [49, 64], [76, 100], [103, 136]]
    let a = make_f32(&[4, 3], (1..=12).map(|x| x as f64).collect());
    let b_in = make_f32(&[3, 2], (1..=6).map(|x| x as f64).collect());
    let expected = make_f32(
        &[4, 2],
        vec![22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0],
    );

    let mut b = Builder::new("test_matmul");
    let lhs = b.input("a", a.tensor_type().elem_type, &[4, 3]);
    let rhs = b.input("b", b_in.tensor_type().elem_type, &[3, 2]);
    let out = b.matmul("m", lhs, rhs);
    b.output(out);

    let got = run_builder(b.graph, &[a, b_in]);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}
