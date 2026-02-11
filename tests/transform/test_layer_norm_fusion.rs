use std::collections::HashMap;

use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
use my_onnx::onnx::model::NodeId;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::model::ValueInfo;
use my_onnx::onnx::operator::args;
use my_onnx::onnx::operator::*;
use my_onnx::tensor::data::ScalarData;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::ResolvedTensorType;
use my_onnx::tensor::types::TensorType;
use my_onnx::tensor::Tensor;
use my_onnx::transform::modify::NodeDelete;
use my_onnx::transform::modify::SimpleGraphOp;
use my_onnx::transform::optimize::layer_norm_fusion::LayerNormFusion;
use my_onnx::transform::Pass;

fn create_value(graph: &mut Graph, name: String, dims: &[usize], elem_type: DataType) -> ValueId {
    let ty = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(dims));
    graph.values.alloc(ValueInfo {
        name,
        ty: Some(TensorType::Resolved(ty)),
    })
}

macro_rules! build_graph {
    (
        name: $name:expr,
        inputs: { $( $input_name:ident : ( $input_ty:expr, $input_dims:expr ) ),* $(,)? },
        outputs: { $( $output_name:ident : ( $output_ty:expr, $output_dims:expr ) ),* $(,)? },
        initializers: { $( $init_name:ident = $init_val:expr ),* $(,)? },
        nodes: [
            $( { $node_name:expr, $op:expr, [ $( $input:ident ),* ] => $output:ident : $out_dims:expr } ),* $(,)?
        ]
    ) => {{
        use std::collections::hash_map::Entry;

        let mut graph = Graph::empty_graph($name.to_string());
        let mut initializers_list = Vec::new();
        let mut registry: HashMap<&str, ValueId> = HashMap::new();

        // Create values
        $(
            let $input_name = create_value(
                &mut graph,
                stringify!($input_name).to_string(),
                $input_dims,
                $input_ty.into(),
            );
            registry.insert(stringify!($input_name), $input_name);

            let node = graph.nodes.alloc(Node::create_node(
                vec![],
                vec![$input_name],
                format!("Input_{}", stringify!($input_name)),
                Operator::Input($input_name),
            ));
            graph.inputs.push(node);
        )*

        $(
            let $output_name = create_value(
                &mut graph,
                stringify!($output_name).to_string(),
                $output_dims,
                $output_ty.into(),
            );
            registry.insert(stringify!($output_name), $output_name);

            let node = graph.nodes.alloc(Node::create_node(
                vec![$output_name],
                vec![],
                format!("Output_{}", stringify!($output_name)),
                Operator::Output($output_name),
            ));
            graph.outputs.push(node);
        )*

        // Create initializers
        $(
            let name = stringify!($init_name);
            let init = graph.values.alloc(ValueInfo {
                name: name.to_string(),
                ty: Some(TensorType::Resolved($init_val.tensor_type())),
            });
            registry.insert(name, init);
            graph.initializer.insert(init, $init_val);
            initializers_list.push(init);
        )*

        // Create nodes
        $(
            let key = stringify!($output);
            let $output = match registry.entry(key) {
                Entry::Occupied(o) => o.get().to_owned(),
                Entry::Vacant(v) => {
                    let val = create_value(
                        &mut graph,
                        stringify!($output).to_string(),
                        $out_dims,
                        DataType::Float(FloatType::F32),
                    );
                    v.insert(val);
                    val
                }
            };

            graph.nodes.alloc(Node::create_node(
                vec![$( registry[stringify!($input)] ),*],
                vec![$output],
                $node_name.to_string(),
                $op,
            ));
        )*

        graph
    }};
}

fn find_nodes<F>(graph: &Graph, predicate: F) -> Vec<NodeId>
where
    F: Fn(&Node) -> bool,
{
    graph
        .nodes
        .iter()
        .filter(|(_, node)| predicate(node))
        .map(|(id, _)| id)
        .collect()
}

fn build_unfused_layer_norm_graph(epsilon: f64) -> Graph {
    let epsilon = ScalarData::Float(FloatType::F32, epsilon).to_tensor_data(1);
    let epsilon = Tensor::new(ResolvedTensorDims::new(&[]), epsilon).unwrap();

    build_graph! {
        name: "layer_norm_test",

        inputs: {
            x: (FloatType::F32, &[2, 10, 768]),
            scale: (FloatType::F32, &[768]),
            bias: (FloatType::F32, &[768]),
        },

        outputs: {
            y: (FloatType::F32, &[2, 10, 768]),
        },

        initializers: {
            epsilon = epsilon,
        },

        nodes: [
            { "Mean", Operator::ReduceMean(Reduce { axes: vec![-1], keepdims: true }),
              [x] => mean_out: &[2, 10, 1] },

            { "D", Operator::Sub,
              [x, mean_out] => d_out: &[2, 10, 768] },

            { "DD", Operator::Mul,
              [d_out, d_out] => dd_out: &[2, 10, 768] },

            { "Var", Operator::ReduceMean(Reduce { axes: vec![-1], keepdims: true }),
              [dd_out] => var_out: &[2, 10, 1] },

            { "VarEps", Operator::Add,
              [var_out, epsilon] => var_eps_out: &[2, 10, 1] },

            { "StdDev", Operator::Sqrt,
              [var_eps_out] => std_out: &[2, 10, 1] },

            { "Normalized", Operator::Div,
              [d_out, std_out] => normalized_out: &[2, 10, 768] },

            { "NormalizedScaled", Operator::Mul,
              [normalized_out, scale] => scaled_out: &[2, 10, 768] },

            { "Y", Operator::Add,
              [scaled_out, bias] => y: &[2, 10, 768] },
        ]
    }
}

#[test]
fn test_valid_pattern_is_fused() {
    let epsilon = 1e-5;
    let mut graph = build_unfused_layer_norm_graph(epsilon);
    let inputs = graph.inputs.clone();
    let outputs = graph.outputs.clone();

    macro_rules! match_input {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Input(id) if id == $value_id)
        }}
    }

    macro_rules! match_output {
        ($value_id:expr, $node_id:expr) => {{
            let node = &graph.nodes[$node_id];
            matches!(node.op, Operator::Output(id) if id == $value_id)
        }}
    }

    let mut modifier = SimpleGraphOp::new(&graph);
    let pass = LayerNormFusion::default();
    pass.run(&mut graph, &mut modifier);
    modifier.update_deleted_nodes(&mut graph);

    let layer_norms = find_nodes(&graph, |node| {
        let eps = match &node.op {
            Operator::LayerNormalization(LayerNormalization { epsilon, .. }) => *epsilon,
            _ => return false,
        };

        eps == epsilon &&
            match_input!(node.inputs[args::LAYER_NORM_DATA], inputs[0]) &&
            match_input!(node.inputs[args::LAYER_NORM_SCALE], inputs[1]) &&
            match_input!(node.inputs[args::LAYER_NORM_BIAS], inputs[2]) &&
            match_output!(node.outputs[0], outputs[0])
    });
    assert_eq!(layer_norms.len(), 1);

    let final_total_nodes = graph.nodes.iter().count();
    assert!(final_total_nodes == 5);
}
