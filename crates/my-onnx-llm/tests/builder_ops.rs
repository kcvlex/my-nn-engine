use std::path::PathBuf;

use my_onnx::onnx::load::LoadProto;
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

fn load_pb(path: PathBuf) -> Tensor {
    Tensor::load_from_path(path).unwrap()
}

fn run_builder(graph: my_onnx::onnx::model::Graph, inputs: &[Tensor]) -> Tensor {
    let opts = Options::builder().target(Target::CPU).build();
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

fn input_for(b: &mut Builder, name: &str, t: &Tensor) -> my_onnx::onnx::model::ValueId {
    let ty = t.tensor_type();
    b.input(
        name,
        ty.elem_type,
        &ty.dims.iter().copied().collect::<Vec<_>>(),
    )
}

fn i64_init(b: &mut Builder, name: &str, values: Vec<i64>) -> my_onnx::onnx::model::ValueId {
    let len = values.len();
    let t = Tensor::new(
        ResolvedTensorDims::new(&[len]),
        TensorData::SInt(SIntType::I64, values),
    )
    .unwrap();
    b.initializer(name, t)
}

#[test]
fn gather_default_axis() {
    let dir = fixture("gather_default_axis");
    let data = load_pb(dir.join("input_0.pb"));
    let indices = load_pb(dir.join("input_1.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let data_ty = data.tensor_type();
    let idx_ty = indices.tensor_type();

    let mut b = Builder::new("test_gather");
    let data_in = b.input(
        "data",
        data_ty.elem_type,
        &data_ty.dims.iter().copied().collect::<Vec<_>>(),
    );
    let idx_in = b.input(
        "indices",
        idx_ty.elem_type,
        &idx_ty.dims.iter().copied().collect::<Vec<_>>(),
    );
    let out = b.gather("g", data_in, idx_in, 0);
    b.output(out);

    let got = run_builder(b.graph, &[data, indices]);
    assert!(got.eq_with_epsilon(&expected, 1e-7, CompPolicy::Either));
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

    let mut b = Builder::new("test_rms_norm");
    let x_in = b.input("x", x.tensor_type().elem_type, &[1, 4]);
    let scale_in = b.initializer("scale", scale);
    let out = b.rms_norm("rms", x_in, scale_in, -1, eps);
    b.output(out);

    let got = run_builder(b.graph, &[x]);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}

#[test]
fn add_via_fixture() {
    let dir = fixture("add");
    let a = load_pb(dir.join("input_0.pb"));
    let b_in = load_pb(dir.join("input_1.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let mut b = Builder::new("test_add");
    let lhs = input_for(&mut b, "a", &a);
    let rhs = input_for(&mut b, "b", &b_in);
    let out = b.add("add", lhs, rhs);
    b.output(out);

    let got = run_builder(b.graph, &[a, b_in]);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}

#[test]
fn mul_basic() {
    let a = make_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_in = make_f32(&[2, 3], vec![2.0, 2.0, 2.0, 0.5, 0.5, 0.5]);
    let expected = make_f32(&[2, 3], vec![2.0, 4.0, 6.0, 2.0, 2.5, 3.0]);

    let mut b = Builder::new("test_mul");
    let lhs = input_for(&mut b, "a", &a);
    let rhs = input_for(&mut b, "b", &b_in);
    let out = b.mul("mul", lhs, rhs);
    b.output(out);

    let got = run_builder(b.graph, &[a, b_in]);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}

#[test]
fn neg_via_fixture() {
    let dir = fixture("neg");
    let x = load_pb(dir.join("input_0.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let mut b = Builder::new("test_neg");
    let x_in = input_for(&mut b, "x", &x);
    let out = b.neg("neg", x_in);
    b.output(out);

    let got = run_builder(b.graph, &[x]);
    assert!(got.eq_with_epsilon(&expected, 1e-5, CompPolicy::Either));
}

#[test]
fn sigmoid_via_fixture() {
    let dir = fixture("sigmoid");
    let x = load_pb(dir.join("input_0.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let mut b = Builder::new("test_sigmoid");
    let x_in = input_for(&mut b, "x", &x);
    let out = b.sigmoid("sigmoid", x_in);
    b.output(out);

    let got = run_builder(b.graph, &[x]);
    assert!(got.eq_with_epsilon(&expected, 1e-6, CompPolicy::Either));
}

#[test]
fn reshape_via_fixture() {
    let dir = fixture("reshape");
    let x = load_pb(dir.join("input_0.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let mut b = Builder::new("test_reshape");
    let x_in = input_for(&mut b, "x", &x);
    let shape = i64_init(&mut b, "shape", vec![3, 1, 1, 2, 4]);
    let out = b.reshape("rs", x_in, shape);
    b.output(out);

    let got = run_builder(b.graph, &[x]);
    assert!(got.eq_with_epsilon(&expected, 1e-7, CompPolicy::Either));
}

#[test]
fn transpose_via_fixture() {
    let dir = fixture("transpose");
    let x = load_pb(dir.join("input_0.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let mut b = Builder::new("test_transpose");
    let x_in = input_for(&mut b, "x", &x);
    let out = b.transpose("tr", x_in, vec![2, 3, 1, 0]);
    b.output(out);

    let got = run_builder(b.graph, &[x]);
    assert!(got.eq_with_epsilon(&expected, 1e-7, CompPolicy::Either));
}

#[test]
fn concat_via_fixture() {
    let dir = fixture("concat_axis_2");
    let a = load_pb(dir.join("input_0.pb"));
    let b_in = load_pb(dir.join("input_1.pb"));
    let c_in = load_pb(dir.join("input_2.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let mut b = Builder::new("test_concat");
    let a_id = input_for(&mut b, "a", &a);
    let b_id = input_for(&mut b, "b", &b_in);
    let c_id = input_for(&mut b, "c", &c_in);
    let out = b.concat("cat", vec![a_id, b_id, c_id], 2);
    b.output(out);

    let got = run_builder(b.graph, &[a, b_in, c_in]);
    assert!(got.eq_with_epsilon(&expected, 1e-7, CompPolicy::Either));
}

#[test]
fn slice_via_fixture() {
    let dir = fixture("slice");
    let x = load_pb(dir.join("input_0.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let mut b = Builder::new("test_slice");
    let x_in = input_for(&mut b, "x", &x);
    let starts = i64_init(&mut b, "starts", vec![1, 0]);
    let ends = i64_init(&mut b, "ends", vec![3, 4]);
    let axes = i64_init(&mut b, "axes", vec![1, 2]);
    let out = b.slice("sl", x_in, starts, ends, Some(axes), None);
    b.output(out);

    let got = run_builder(b.graph, &[x]);
    assert!(got.eq_with_epsilon(&expected, 1e-7, CompPolicy::Either));
}

#[test]
fn expand_via_fixture() {
    let dir = fixture("expand");
    let x = load_pb(dir.join("input_0.pb"));
    let expected = load_pb(dir.join("output_0.pb"));

    let mut b = Builder::new("test_expand");
    let x_in = input_for(&mut b, "x", &x);
    let shape = i64_init(&mut b, "shape", vec![3, 4]);
    let out = b.expand("ex", x_in, shape);
    b.output(out);

    let got = run_builder(b.graph, &[x]);
    assert!(got.eq_with_epsilon(&expected, 1e-7, CompPolicy::Either));
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
