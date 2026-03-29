use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::transform::modify::GraphOp;
use crate::transform::utils::*;
use crate::transform::Pass;

#[derive(Default)]
pub struct DecomposeConv {}
#[derive(Default)]
pub struct DecomposeMaxPool {}

fn gen_im2col_from_conv(
    conv: &Conv,
    kernel_shape: &ResolvedTensorDims,
    output_shape: &ResolvedTensorDims,
) -> (Im2Col, ResolvedTensorDims) {
    let channel = kernel_shape[1];
    let nbatch = output_shape[0];

    // Drop nbatch and feature_map_count
    let one_fm_shape: ResolvedTensorDims = output_shape.iter().skip(2).copied().collect();

    let one_kernel_shape: ResolvedTensorDims = kernel_shape.iter().skip(2).copied().collect();

    let one_fm_size = one_fm_shape.size(); // size of one feature_map
    let one_kernel_size = one_kernel_shape.size();
    let im2col_output_shape =
        ResolvedTensorDims::new(&[one_fm_size * nbatch, one_kernel_size * channel]);
    let im2col = Im2Col {
        nbatch,
        one_fm_shape,
        pad: conv.pad.clone(),
        channel: Channel::Meld(channel),
        dilations: conv.dilations.clone(),
        one_kernel_shape,
        strides: conv.strides.clone(),
        pad_val: PadVal::Zero,
    };
    (im2col, im2col_output_shape)
}

fn gen_im2col_from_pooling(
    pooling: &Pooling,
    output_shape: &ResolvedTensorDims,
) -> (Im2Col, ResolvedTensorDims) {
    let kernel_shape = &pooling.kernel_shape;
    let nbatch = output_shape[0];
    let channel = output_shape[1];

    // Drop nbatch and channel
    let one_fm_shape: ResolvedTensorDims = output_shape.iter().skip(2).copied().collect();

    let row = one_fm_shape.size() * channel * nbatch;
    let col = kernel_shape.size();
    let im2col_output_shape = ResolvedTensorDims::new(&[row, col]);
    let im2col = Im2Col {
        nbatch,
        one_fm_shape,
        pad: pooling.pad.clone(),
        channel: Channel::Split(channel),
        dilations: pooling.dilations.clone(),
        one_kernel_shape: kernel_shape.clone(),
        strides: pooling.strides.clone(),
        pad_val: PadVal::NInf,
    };
    (im2col, im2col_output_shape)
}

fn im2col_core<T: GraphOp>(graph: &mut Graph, modifier: &mut T, id: NodeId) {
    let index = id.index();
    let node = &graph.nodes[id];
    match &node.op {
        Operator::Conv(ref conv) => {
            let conv = conv.clone();
            let data_value = node.inputs[args::CONV_DATA];
            let kernel_value = node.inputs[args::CONV_WEIGHT];
            let bias_value = node.inputs.get(args::CONV_BIAS).copied();
            let old_output_value = node.outputs[0];
            let kernel = graph
                .get_resolved_tensor_type(kernel_value)
                .unwrap()
                .clone();
            let kernel_shape = &kernel.dims;
            let output_dims = graph
                .get_resolved_tensor_type(old_output_value)
                .unwrap()
                .dims
                .clone();
            let nchw_output_shape = match conv.output_layout {
                Layout::NCHW => output_dims,
                Layout::NHWC => output_dims.transpose(&[0, 3, 1, 2]),
            };
            let (im2col, im2col_output_shape) =
                gen_im2col_from_conv(&conv, kernel_shape, &nchw_output_shape);
            let nbatch = im2col.nbatch;
            let one_fm_shape = im2col.one_fm_shape.clone();
            let feature_map_count = kernel_shape[0];

            // Im2Col the input data
            let im2col_data = modifier.register_new_value(
                graph,
                format!("Im2Col_{index}_ExpandedData"),
                ResolvedTensorType::new(kernel.elem_type, im2col_output_shape.clone()),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![data_value],
                    outputs: vec![im2col_data],
                    name: format!("Im2Col_{index}"),
                    op: Operator::Im2Col(im2col),
                    meta: NodeMeta::default(),
                },
            );

            // DEBUG
            //
            // let force_reshape_output = modifier.register_new_value(
            //     graph,
            //     format!("Im2Col_{index}_ForceReshapeOutput"),
            //     ResolvedTensorType::new(
            //         kernel.elem_type,
            //         im2col_output_shape.clone(),
            //     ),
            // );
            // modifier.register_new_node(
            //     graph,
            //     Node {
            //         inputs: vec![im2col_data],
            //         outputs: vec![force_reshape_output],
            //         name: format!("Im2Col_{index}_ForceReshape"),
            //         op: Operator::ForceReshape,
            //     },
            // );
            // modifier.replace_input_value(graph, old_output_value, force_reshape_output);

            // Reshape the kernel
            let reshaped_kernel = ReshapeGenerator::default()
                .set_input(kernel_value)
                .set_dims(&[feature_map_count, kernel_shape.size() / feature_map_count])
                .set_node_name(format!("Im2Col_{index}_ReshapeKernel"))
                .set_value_name(format!("Im2Col_{index}_ReshapeKernel"))
                .generate(graph, modifier)
                .unwrap();

            let dims = ResolvedTensorDims::new(&[
                im2col_output_shape[0],
                kernel_shape.size() / im2col_output_shape[1],
            ]);
            assert_eq!(dims.size(), nchw_output_shape.size());
            // Gemm
            let gemm_output = modifier.register_new_value(
                graph,
                format!("Im2Col_{index}_GemmOutput"),
                ResolvedTensorType::new(kernel.elem_type, dims),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![im2col_data, reshaped_kernel],
                    outputs: vec![gemm_output],
                    name: format!("Im2Col_{index}_Gemm"),
                    op: Operator::Gemm(Gemm {
                        trans_a: false,
                        trans_b: true,
                        alpha: 1.0,
                        beta: 0.0,
                    }),
                    meta: NodeMeta::default(),
                },
            );

            // Add bias right after Gemm (before Reshape/Transpose)
            // bias is 1D [C_out] which broadcasts with Gemm output [M, C_out]
            let biased_output = if let Some(bias) = bias_value {
                let add_output = modifier.register_new_value(
                    graph,
                    format!("Im2Col_{index}_AddBias"),
                    graph.get_resolved_tensor_type(gemm_output).unwrap().clone(),
                );
                modifier.register_new_node(
                    graph,
                    Node {
                        inputs: vec![gemm_output, bias],
                        outputs: vec![add_output],
                        name: format!("Im2Col_{index}_AddBias"),
                        op: Operator::Add,
                        meta: NodeMeta::default(),
                    },
                );
                add_output
            } else {
                gemm_output
            };

            // Reshape the output
            let reshaped_output_shape = {
                let mut vec = Vec::with_capacity(one_fm_shape.ndim() + 2);
                vec.push(nbatch);
                vec.extend(one_fm_shape.iter().copied());
                vec.push(feature_map_count);
                vec
            };
            let reshaped_output = ReshapeGenerator::default()
                .set_input(biased_output)
                .set_dims(&reshaped_output_shape)
                .set_node_name(format!("Im2Col_{index}_ReshapeOutput"))
                .set_value_name(format!("Im2Col_{index}_ReshapeOutput"))
                .generate(graph, modifier)
                .unwrap();

            // Transpose the output.
            let perm = {
                let mut vec = Vec::with_capacity(one_fm_shape.ndim() + 2);
                vec.push(0);
                vec.push(one_fm_shape.ndim() + 1);
                vec.extend((0..one_fm_shape.ndim()).map(|x| x + 1));
                vec
            };

            // TODO: Avoid to force contiguous after the feature for the replacement with a
            // different shape (strides) is supported.
            let transposed_output = TransposeGenerator::default()
                .set_contiguous(true)
                .set_input(reshaped_output)
                .set_perm(perm)
                .set_node_name(format!("Im2Col_{index}_TransposeOutput"))
                .set_value_name(format!("Im2Col_{index}_TransposeOutput"))
                .generate(graph, modifier)
                .unwrap();

            modifier.replace_input_value(graph, old_output_value, transposed_output);
        }

        Operator::MaxPool(ref pooling) => {
            let data_value = node.inputs[args::MAXPOOL_DATA];
            let old_output_value = node.outputs[0];

            let (elem_type, output_shape) = &graph
                .get_resolved_tensor_type(old_output_value)
                .map(|x| (x.elem_type, x.dims.clone()))
                .unwrap();

            let op = match &node.op {
                Operator::MaxPool(_) => ReduceOp::Max,
                _ => unreachable!(),
            };

            let (im2col, im2col_output_shape) = gen_im2col_from_pooling(pooling, output_shape);
            let row = im2col_output_shape[0];

            // Im2Col
            let im2col_data = modifier.register_new_value(
                graph,
                format!("Im2Col_{index}_ExpandedData"),
                ResolvedTensorType::new(*elem_type, im2col_output_shape),
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![data_value],
                    outputs: vec![im2col_data],
                    name: format!("Im2Col_{index}"),
                    op: Operator::Im2Col(im2col),
                    meta: NodeMeta::default(),
                },
            );

            // Reduce
            let dims = ResolvedTensorDims::new(&[row]);
            assert_eq!(dims.size(), output_shape.size());
            let reduced_data = modifier.register_new_value(
                graph,
                format!("Im2Col_{index}_ReducedData"),
                ResolvedTensorType::new(*elem_type, dims),
            );
            let reduce_node = Node {
                inputs: vec![im2col_data],
                outputs: vec![reduced_data],
                name: format!("Im2Col_{index}_Reduce"),
                op: Operator::ReduceMatrix(op),
                meta: NodeMeta::default(),
            };
            modifier.register_new_node(graph, reduce_node);

            // Reshape
            let reshaped_output = ReshapeGenerator::default()
                .set_input(reduced_data)
                .set_dims(&output_shape[..])
                .set_node_name(format!("Im2Col_{index}_ReshapeOutput"))
                .set_value_name(format!("Im2Col_{index}_ReshapeOutput"))
                .generate(graph, modifier)
                .unwrap();

            modifier.replace_input_value(graph, old_output_value, reshaped_output);
        }
        _ => unreachable!(),
    }
}

impl<T: GraphOp> Pass<T> for DecomposeConv {
    fn summary(&self) -> &'static str {
        "Decompose Conv into Im2Col + Gemm"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let res = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if matches!(node.op, Operator::Conv(_)) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for id in res.into_iter() {
            im2col_core(graph, modifier, id);
        }
    }
}

impl<T: GraphOp> Pass<T> for DecomposeMaxPool {
    fn summary(&self) -> &'static str {
        "Decompose MaxPool into Im2Col + Reduce"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let res = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if matches!(node.op, Operator::MaxPool(_)) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for id in res.into_iter() {
            im2col_core(graph, modifier, id);
        }
    }
}
