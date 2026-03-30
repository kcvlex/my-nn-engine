mod contiguous_insertion;
mod decomposition;
mod nhwc2nchw_detection;
mod nhwc2nchw_inertion;
mod nhwc2nchw_sink_and_fold;
pub mod strides;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::options::*;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::transform::lower::contiguous_insertion::ContiguousInsertion;
use crate::transform::lower::decomposition::AttentionDecomposition;
use crate::transform::lower::decomposition::ConvDecomposition;
use crate::transform::lower::decomposition::MaxPoolDecomposition;
use crate::transform::lower::decomposition::ReduceDecomposition;
use crate::transform::lower::nhwc2nchw_detection::NHWC2NCHWDetection;
use crate::transform::lower::nhwc2nchw_inertion::NHWC2NCHWInsertion;
use crate::transform::lower::nhwc2nchw_sink_and_fold::NHWC2NCHWSinkAndFold;
use crate::transform::lower::strides::AssignStrides;
use crate::transform::modify::GraphOp;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::shape::verify::ShapeVerification;
use crate::transform::utils::ContiguousElimination;
use crate::transform::utils::ContiguousFolding;
use crate::transform::utils::ReinterpretConversion;
use crate::transform::utils::ReshapeGenerator;
use crate::transform::Pass;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

#[derive(Default)]
pub struct GlobalAvgPoolElimination {}

impl<T: GraphOp> Pass<T> for GlobalAvgPoolElimination {
    fn summary(&self) -> &'static str {
        "Convert GlobalAveragePool to another operator"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if let Operator::GlobalAveragePool = node.op {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for id in ids.iter() {
            let input = graph.nodes[*id].inputs[0];
            let old_output = graph.nodes[*id].outputs[0];
            let input_ty = graph.get_resolved_tensor_type(input).unwrap().clone();
            let nbatch = input_ty.dims[0];
            let channel = input_ty.dims[1];
            let row = nbatch * channel;
            let col = input_ty.dims.size() / row;
            let elem_ty = input_ty.elem_type;

            let reshaped = ReshapeGenerator::default()
                .set_input(input)
                .set_dims(&[row, col])
                .set_node_name(format!("GlobalAveragePool_Reshaped_{}", input.index()))
                .set_value_name(format!("GlobalAveragePool_Reshaped_{}", input.index()))
                .generate(graph, modifier)
                .unwrap();

            let pool_output = modifier.register_new_value(
                graph,
                format!("GlobalAveragePool_Output_{}", id.index()),
                ResolvedTensorType::new(elem_ty, ResolvedTensorDims::new(&[row, 1])),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![reshaped],
                    outputs: vec![pool_output],
                    op: Operator::ReduceMatrix(ReduceOp::Mean),
                    name: format!("GlobalAveragePool_{}", id.index()),
                    meta: NodeMeta::default(),
                },
            );

            let mut new_dims = vec![
                1;
                graph
                    .get_resolved_tensor_type(old_output)
                    .unwrap()
                    .dims
                    .ndim()
            ];
            new_dims[0] = nbatch;
            new_dims[1] = channel;
            let new_output = ReshapeGenerator::default()
                .set_input(pool_output)
                .set_dims(&new_dims)
                .set_node_name(format!(
                    "GlobalAveragePool_Reshaped_{}",
                    pool_output.index()
                ))
                .set_value_name(format!(
                    "GlobalAveragePool_Reshaped_{}",
                    pool_output.index()
                ))
                .generate(graph, modifier)
                .unwrap();

            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

pub fn create_lower_passes(opt: &Options, enable_nhwc: bool) -> SimplePassManager<SimpleGraphOp> {
    let mut passes = SimplePassManager::new("Lowering".to_string());
    passes.add_pass(Box::new(ReduceDecomposition::default()));
    passes.add_pass(Box::new(GlobalAvgPoolElimination::default()));
    passes.add_pass(Box::new(ContiguousInsertion::default()));
    if enable_nhwc {
        passes.add_pass(Box::new(NHWC2NCHWInsertion::default()));
    }
    passes.add_pass(Box::new(AssignStrides { target: opt.target }));
    passes.add_pass(Box::new(ShapeVerification {
        target: opt.target,
        check_strides: true,
    }));

    // NHWC optimization
    if enable_nhwc {
        passes.add_pass(Box::new(ReinterpretConversion::default()));
        passes.add_pass(Box::new(ContiguousFolding::backward_only()));
        passes.add_pass(Box::new(ContiguousElimination::default()));
        passes.add_pass(Box::new(ShapeVerification {
            target: opt.target,
            check_strides: true,
        }));
        passes.add_pass(Box::new(NHWC2NCHWDetection::default()));
        passes.add_pass(Box::new(ShapeVerification {
            target: opt.target,
            check_strides: true,
        }));
        passes.add_pass(Box::new(NHWC2NCHWSinkAndFold::default()));
        passes.add_pass(Box::new(ShapeVerification {
            target: opt.target,
            check_strides: true,
        }));
    }

    // Decompose after layout
    if matches!(opt.target, Target::CPU) {
        passes.add_pass(Box::new(AttentionDecomposition::default()));
        passes.add_pass(Box::new(ConvDecomposition::default()));
        passes.add_pass(Box::new(MaxPoolDecomposition::default()));
    }
    passes
}
