use std::path::PathBuf;

use my_onnx::onnx::load::LoadProto;
use my_onnx::onnx::model::Model;
use my_onnx::onnx::operator::Operator;
use my_onnx::options::Options;
use my_onnx::tensor::Tensor;
use my_onnx::transform::transform_graph;

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

    transform_graph(&mut model.graph, &Options::builder().build());
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
