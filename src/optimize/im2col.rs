use crate::model::{Graph, Node, ValueId};
use crate::operator::*;
use crate::optimize::optimizer::{GraphModifier, Pass};
use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use crate::tensor::tensor::ResolvedTensorType;

#[derive(Default)]
pub struct InsertIm2Col {}

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

    let one_fm_size = output_shape.size(); // size of one feature_map
    let one_kernel_size = one_kernel_shape.size();
    let im2col_output_shape =
        ResolvedTensorDims::new(vec![one_fm_size * nbatch, one_kernel_size * channel]);
    let im2col = Im2Col {
        nbatch,
        one_fm_shape,
        pad: conv.pad.clone(),
        channel: Channel::Meld(channel),
        dilations: conv.dilations.clone(),
        one_kernel_shape,
        strides: conv.strides.clone(),
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
    let im2col_output_shape = ResolvedTensorDims::new(vec![row, col]);
    let im2col = Im2Col {
        nbatch,
        one_fm_shape,
        pad: pooling.pad.clone(),
        channel: Channel::Split(channel),
        dilations: pooling.dilations.clone(),
        one_kernel_shape: kernel_shape.clone(),
        strides: pooling.strides.clone(),
    };
    (im2col, im2col_output_shape)
}

impl<T: GraphModifier> Pass<T> for InsertIm2Col {
    fn summary(&self) -> &'static str {
        "Insert explicit Im2Col nodes and expand Conv/MaxPool"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let res = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if matches!(node.op, Operator::Conv(_) | Operator::MaxPool(_)) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for (index, id) in res.into_iter().enumerate() {
            let node = &graph.nodes[id];
            match &node.op {
                Operator::Conv(ref conv) => {
                    let data_value = node.inputs[args::CONV_DATA];
                    let kernel_value = node.inputs[args::CONV_WEIGHT];
                    let old_output_value = node.outputs[0];
                    let kernel = graph
                        .get_resolved_tensor_type(kernel_value)
                        .unwrap()
                        .clone();
                    let kernel_shape = &kernel.dims;
                    let output_shape = &graph
                        .get_resolved_tensor_type(old_output_value)
                        .unwrap()
                        .dims;
                    let (im2col, im2col_output_shape) =
                        gen_im2col_from_conv(conv, kernel_shape, output_shape);
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
                            mark_as_deleted: false,
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
                    let reshaped_kernel = modifier.register_new_value(
                        graph,
                        format!("Im2Col_{index}_ReshapedKernel"),
                        ResolvedTensorType::new(
                            kernel.elem_type,
                            ResolvedTensorDims::new(vec![
                                feature_map_count,
                                kernel_shape.size() / feature_map_count,
                            ]),
                        ),
                    );
                    modifier.register_new_node(
                        graph,
                        Node {
                            inputs: vec![kernel_value],
                            outputs: vec![reshaped_kernel],
                            name: format!("Im2Col_{index}_ReshapeKernel"),
                            op: Operator::Reshape,
                            mark_as_deleted: false,
                        },
                    );
                    // Gemm
                    let gemm_output = modifier.register_new_value(
                        graph,
                        format!("Im2Col_{index}_GemmOutput"),
                        ResolvedTensorType::new(
                            kernel.elem_type,
                            ResolvedTensorDims::new(vec![
                                im2col_output_shape[0],
                                kernel_shape.size() / im2col_output_shape[1],
                            ]),
                        ),
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
                                trans_c: false,
                                alpha: 1.0,
                                beta: 0.0,
                            }),
                            mark_as_deleted: false,
                        },
                    );

                    // Reshape the output
                    let reshaped_output_shape = {
                        let mut vec = Vec::with_capacity(one_fm_shape.ndim() + 2);
                        vec.push(nbatch);
                        vec.extend(one_fm_shape.iter().copied());
                        vec.push(feature_map_count);
                        vec
                    };
                    let reshaped_output = modifier.register_new_value(
                        graph,
                        format!("Im2Col_{index}_ReshapedOutput"),
                        ResolvedTensorType::new(
                            kernel.elem_type,
                            ResolvedTensorDims::new(reshaped_output_shape.clone()),
                        ),
                    );
                    modifier.register_new_node(
                        graph,
                        Node {
                            inputs: vec![gemm_output],
                            outputs: vec![reshaped_output],
                            name: format!("Im2Col_{index}_ReshapeOutput"),
                            op: Operator::Reshape,
                            mark_as_deleted: false,
                        },
                    );

                    // Transpose the output
                    let perm = {
                        let mut vec = Vec::with_capacity(one_fm_shape.ndim() + 2);
                        vec.push(0);
                        vec.push(one_fm_shape.ndim() + 1);
                        vec.extend((0..one_fm_shape.ndim()).map(|x| x + 1));
                        vec
                    };
                    let tranposed_output_shape =
                        perm.iter().map(|i| reshaped_output_shape[*i]).collect();
                    let transposed_output = modifier.register_new_value(
                        graph,
                        format!("Im2Col_{index}_TransposedOutput"),
                        ResolvedTensorType::new(
                            kernel.elem_type,
                            ResolvedTensorDims::new(tranposed_output_shape),
                        ),
                    );
                    modifier.register_new_node(
                        graph,
                        Node {
                            inputs: vec![reshaped_output],
                            outputs: vec![transposed_output],
                            name: format!("Im2Col_{index}_TransposeOutput"),
                            op: Operator::Transpose(perm),
                            mark_as_deleted: false,
                        },
                    );

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

                    let (im2col, im2col_output_shape) =
                        gen_im2col_from_pooling(pooling, output_shape);
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
                            mark_as_deleted: false,
                        },
                    );

                    // Reduce
                    let reduced_data = modifier.register_new_value(
                        graph,
                        format!("Im2Col_{index}_ReducedData"),
                        ResolvedTensorType::new(*elem_type, vec![row].into()),
                    );
                    let reduce_node = Node {
                        inputs: vec![im2col_data],
                        outputs: vec![reduced_data],
                        name: format!("Im2Col_{index}_Reduce"),
                        op: Operator::ReduceMatrix(op),
                        mark_as_deleted: false,
                    };
                    modifier.register_new_node(graph, reduce_node);

                    // Reshape
                    let reshaped_output = modifier.register_new_value(
                        graph,
                        format!("Im2Col_{index}_ReshapedOutput"),
                        ResolvedTensorType::new(*elem_type, output_shape.clone()),
                    );
                    modifier.register_new_node(
                        graph,
                        Node {
                            inputs: vec![reduced_data],
                            outputs: vec![reshaped_output],
                            name: format!("Im2Col_{index}_ReshapeOutput"),
                            op: Operator::Reshape,
                            mark_as_deleted: false,
                        },
                    );
                    modifier.replace_input_value(graph, old_output_value, reshaped_output);
                }
                _ => unreachable!(),
            }
        }
    }
}
