use std::path::PathBuf;

use my_onnx::onnx::load::LoadProto;
use my_onnx::onnx::model::Model;
use my_onnx::onnx::operator::Layout;
use my_onnx::onnx::operator::Operator;
use my_onnx::options::*;
use my_onnx::tensor::Tensor;
use my_onnx::transform::transform_graph;

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

    transform_graph(&mut model.graph, &Options::builder().target(target).build());
    model
}

fn load_and_transform(model_dir: &str, model_file: &str, num_inputs: usize) -> Model {
    load_and_transform_with_target(model_dir, model_file, num_inputs, Target::CUDA)
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

#[test]
fn test_resnet18_nhwc_layout_cpu() {
    let model =
        load_and_transform_with_target("resnet18-v2-7", "resnet18-v2-7.onnx", 1, Target::CPU);

    // All NHWC2NCHW should be lowered away
    let nhwc2nchw = count_op(&model, |op| matches!(op, Operator::NHWC2NCHW));
    assert_eq!(nhwc2nchw, 0);

    // Currently SinkNHWC2NCHW cannot pass through multi-input ops (BN, Add),
    // so no Im2Col gets NHWC layout yet in ResNet v2 (BN -> Relu -> Conv).
    // This test documents the current state.
    let nhwc_im2col = model
        .graph
        .nodes
        .iter()
        .filter(|(_, node)| match &node.op {
            Operator::Im2Col(im2col) => im2col.layout == Layout::NHWC,
            _ => false,
        })
        .count();
    let total_im2col = count_op(&model, |op| matches!(op, Operator::Im2Col(_)));
    eprintln!("Im2Col: {nhwc_im2col}/{total_im2col} NHWC");
}
