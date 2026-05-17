#![allow(dead_code, unused_imports)]

use std::path::PathBuf;

use my_nn_engine::graph::Graph;
use my_nn_engine::graph::Node;
use my_nn_engine::graph::NodeId;
use my_nn_engine::graph::ValueId;
use my_nn_engine::graph::ValueInfo;
use my_nn_engine::onnx::load::*;
use my_nn_engine::options::*;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::data::TensorData;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::ResolvedTensorType;
use my_nn_engine::tensor::types::TensorType;
use my_nn_engine::tensor::Tensor;

pub fn find_nodes<F>(graph: &Graph, predicate: F) -> Vec<NodeId>
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

pub fn make_1d_tensor(data: Vec<f64>) -> Tensor {
    let len = data.len();
    Tensor::new(
        ResolvedTensorDims::new(&[len]),
        TensorData::Float(FloatType::F32, data),
    )
    .unwrap()
}

pub fn make_tensor(dims: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::new(dims),
        TensorData::Float(FloatType::F32, data),
    )
    .unwrap()
}

pub fn run_validated_model(
    model: &str,
    epsilon: f64,
    nums: (usize, usize),
    model_filename: Option<&str>,
    options: Options,
) -> std::result::Result<(), SessionError> {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model);
    let model_path = root_dir.join(model_filename.unwrap_or(format!("{model}.onnx").as_str()));

    let data_dirs: Vec<_> = (0..)
        .map(|i| root_dir.join(format!("test_data_set_{i}")))
        .take_while(|p| p.exists())
        .collect();
    assert!(
        !data_dirs.is_empty(),
        "No test_data_set_* found for {model}"
    );

    let (num_inputs, num_outputs) = nums;

    let first_inputs = (0..num_inputs)
        .map(|i| {
            Tensor::load_from_path(data_dirs[0].join(format!("input_{i}.pb")))
                .map_err(SessionError::ModelLoadError)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let input_types = first_inputs
        .iter()
        .map(|input| input.tensor_type())
        .collect::<Vec<_>>();
    let mut session = Session::new(
        &model_path,
        Some(&input_types),
        &options,
        &SessionConfig::default(),
    )?;

    for (set_idx, data_dir) in data_dirs.iter().enumerate() {
        let inputs = if set_idx == 0 {
            first_inputs.clone()
        } else {
            (0..num_inputs)
                .map(|i| {
                    Tensor::load_from_path(data_dir.join(format!("input_{i}.pb")))
                        .map_err(SessionError::ModelLoadError)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        let outputs = session.run(&inputs)?;
        let expected = (0..num_outputs)
            .map(|i| {
                Tensor::load_from_path(data_dir.join(format!("output_{i}.pb")))
                    .map_err(SessionError::ModelLoadError)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (output, expected) in outputs.iter().zip(expected.iter()) {
            assert!(
                output.eq_with_epsilon(expected, epsilon, CompPolicy::Either),
                "Output mismatch in test_data_set_{set_idx} (epsilon={epsilon})",
            );
        }
    }
    Ok(())
}

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
                vec![Some($output_name)],
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
            graph.set_initializer(init, $init_val);
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
                vec![$( Some(registry[stringify!($input)]) ),*],
                vec![$output],
                $node_name.to_string(),
                $op,
            ));
        )*

        graph
    }};
}
