use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::model::ValueInfo;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::ResolvedTensorType;
use my_onnx::tensor::types::TensorType;

pub fn create_value(
    graph: &mut Graph,
    name: String,
    dims: &[usize],
    elem_type: DataType,
) -> ValueId {
    let ty = ResolvedTensorType::new(elem_type, ResolvedTensorDims::new(dims));
    graph.values.alloc(ValueInfo {
        name,
        ty: Some(TensorType::Resolved(ty)),
    })
}

#[macro_export]
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
