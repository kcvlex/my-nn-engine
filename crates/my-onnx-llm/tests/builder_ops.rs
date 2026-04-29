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
