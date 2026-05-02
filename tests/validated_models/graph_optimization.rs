use std::path::PathBuf;

use my_nn_engine::onnx::load::LoadProto;
use my_nn_engine::onnx::model::Model;
use my_nn_engine::onnx::operator::Layout;
use my_nn_engine::onnx::operator::Operator;
use my_nn_engine::options::*;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::tensor::Tensor;
use my_nn_engine::transform::transform_graph;

fn load_and_transform(model_dir: &str, model_file: &str, num_inputs: usize) -> Model {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model_dir);
    let mut model = Model::load_from_path(root.join(model_file)).unwrap();

    let data_dir = root.join("test_data_set_0");
    let input_types: Vec<_> = (0..num_inputs)
        .map(|i| {
            Tensor::load_from_path(data_dir.join(format!("input_{i}.pb")))
                .unwrap()
                .tensor_type()
        })
        .collect();
    model.graph.resolve_input_types(&input_types).unwrap();

    transform_graph(
        &mut model.graph,
        &Options::builder().target(Target::CUDA).build(),
        &SessionConfig::default(),
    );
    model
}

fn count_op(model: &Model, pred: fn(&Operator) -> bool) -> usize {
    model
        .graph
        .nodes
        .iter()
        .filter(|(_, node)| pred(&node.op))
        .count()
}

#[test]
fn test_bert_graph_optimization() {
    let model = load_and_transform("bertsquad-12", "bertsquad-12.onnx", 4);

    let layer_norm = count_op(&model, |op| matches!(op, Operator::LayerNormalization(_)));
    let attention = count_op(&model, |op| matches!(op, Operator::Attention(_)));
    let gelu = count_op(&model, |op| matches!(op, Operator::GeLU(_)));
    let tanh = count_op(&model, |op| matches!(op, Operator::Tanh));
    let reduce_mean = count_op(&model, |op| matches!(op, Operator::ReduceMean(_)));

    // 12 encoder layers × 2 LayerNorms + 1 embedding LayerNorm = 25
    assert_eq!(layer_norm, 25);
    // 12 attention heads
    assert_eq!(attention, 12);
    // 12 intermediate GeLU activations
    assert_eq!(gelu, 12);
    // All Tanh should be absorbed into GeLU
    assert_eq!(tanh, 0);
    // All ReduceMean should be absorbed into LayerNormalization
    assert_eq!(reduce_mean, 0);
}

#[test]
fn test_gpt2_graph_optimization() {
    let model = load_and_transform("GPT2", "model.onnx", 1);

    let layer_norm = count_op(&model, |op| matches!(op, Operator::LayerNormalization(_)));
    let attention = count_op(&model, |op| matches!(op, Operator::Attention(_)));
    let gelu = count_op(&model, |op| matches!(op, Operator::GeLU(_)));
    let tanh = count_op(&model, |op| matches!(op, Operator::Tanh));
    let reduce_mean = count_op(&model, |op| matches!(op, Operator::ReduceMean(_)));

    // 12 layers × 2 LayerNorms + 1 final LayerNorm = 25
    assert_eq!(layer_norm, 25);
    // 12 attention heads
    assert_eq!(attention, 12);
    // 12 intermediate GeLU activations
    assert_eq!(gelu, 12);
    assert_eq!(tanh, 0);
    assert_eq!(reduce_mean, 0);
}

fn load_and_transform_with_target(
    model_dir: &str,
    model_file: &str,
    num_inputs: usize,
    target: Target,
) -> Model {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model_dir);
    let mut model = Model::load_from_path(root.join(model_file)).unwrap();

    let data_dir = root.join("test_data_set_0");
    let input_types: Vec<_> = (0..num_inputs)
        .map(|i| {
            Tensor::load_from_path(data_dir.join(format!("input_{i}.pb")))
                .unwrap()
                .tensor_type()
        })
        .collect();
    model.graph.resolve_input_types(&input_types).unwrap();

    transform_graph(
        &mut model.graph,
        &Options::builder().target(target).build(),
        &SessionConfig::default(),
    );
    model
}

#[cfg(feature = "local")]
#[test]
fn test_tinyllama_graph_optimization() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/hf/tinyllama");
    let mut model = Model::load_from_path(root.join("model.onnx")).unwrap();

    let data_dir = root.join("test_data_set_0");
    let mut input_types = Vec::new();
    for i in 0.. {
        let p = data_dir.join(format!("input_{i}.pb"));
        if !p.exists() {
            break;
        }
        input_types.push(Tensor::load_from_path(p).unwrap().tensor_type());
    }
    model.graph.resolve_input_types(&input_types).unwrap();
    transform_graph(
        &mut model.graph,
        &Options::builder().target(Target::CUDA).build(),
        &SessionConfig::default(),
    );

    let rms_norm = count_op(&model, |op| matches!(op, Operator::RMSNormalization(_)));
    let reduce_mean = count_op(&model, |op| matches!(op, Operator::ReduceMean(_)));
    let pow = count_op(&model, |op| matches!(op, Operator::Pow));
    let sqrt = count_op(&model, |op| matches!(op, Operator::Sqrt));

    // 22 transformer layers × 2 RMSNorms (input_layernorm + post_attention_layernorm)
    //   + 1 final norm = 45
    assert_eq!(rms_norm, 45);
    // All ReduceMean / Pow / Sqrt should be absorbed into RMSNormalization
    assert_eq!(reduce_mean, 0);
    assert_eq!(pow, 0);
    assert_eq!(sqrt, 0);
}

#[test]
fn test_resnet18_nhwc_sink_cuda() {
    let model =
        load_and_transform_with_target("resnet18-v2-7", "resnet18-v2-7.onnx", 1, Target::CUDA);

    let total_conv = count_op(&model, |op| matches!(op, Operator::Conv(_)));
    let nhwc_input_conv = model
        .graph
        .nodes
        .iter()
        .filter(|(_, node)| match &node.op {
            Operator::Conv(conv) => conv.input_layout == Layout::NHWC,
            _ => false,
        })
        .count();
    let nhwc_output_conv = model
        .graph
        .nodes
        .iter()
        .filter(|(_, node)| match &node.op {
            Operator::Conv(conv) => conv.output_layout == Layout::NHWC,
            _ => false,
        })
        .count();
    let nhwc2nchw = count_op(&model, |op| matches!(op, Operator::NHWC2NCHW));

    assert_eq!(total_conv, 20);
    assert_eq!(nhwc_input_conv, 18);
    assert_eq!(nhwc_output_conv, 17);
    assert_eq!(nhwc2nchw, 0);
}
