use std::sync::Arc;

use my_nn_engine::collective;
use my_nn_engine::graph::operator::Operator;
use my_nn_engine::graph::Graph;
use my_nn_engine::graph::Node;
use my_nn_engine::graph::ValueInfo;
use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::data::TensorData;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::ResolvedTensorDims;
use my_nn_engine::tensor::types::ResolvedTensorType;
use my_nn_engine::tensor::types::TensorType;
use my_nn_engine::tensor::Tensor;
use my_nn_engine_comm::UdsCommunicator;

fn build_all_reduce_graph(num_elements: usize) -> Graph {
    let mut graph = Graph::empty_graph("all_reduce_test".to_string());
    let x = graph.values.alloc(ValueInfo {
        name: "x".to_string(),
        ty: Some(TensorType::Resolved(ResolvedTensorType::new(
            DataType::Float(FloatType::F32),
            ResolvedTensorDims::new(&[num_elements]),
        ))),
    });
    let input_node = graph.nodes.alloc(Node::create_node(
        vec![],
        vec![x],
        "Input_x".to_string(),
        Operator::Input(x),
    ));
    graph.inputs.push(input_node);

    let y = graph.values.alloc(ValueInfo {
        name: "y".to_string(),
        ty: None,
    });
    graph.nodes.alloc(Node::create_node(
        vec![Some(x)],
        vec![y],
        "all_reduce".to_string(),
        Operator::AllReduce,
    ));

    let output_node = graph.nodes.alloc(Node::create_node(
        vec![Some(y)],
        vec![],
        "Output_y".to_string(),
        Operator::Output(y),
    ));
    graph.outputs.push(output_node);
    graph
}

#[test]
fn all_reduce_world_size_one_is_identity() {
    let dir = tempfile::tempdir().unwrap();
    let comm = UdsCommunicator::connect(&dir.path().join("comm.sock"), 0, 1).unwrap();
    collective::set_communicator(Arc::new(comm));

    let opt = Options::builder().target(Target::CPU).build();
    let mut session =
        Session::from_graph(build_all_reduce_graph(4), &opt, &SessionConfig::default()).unwrap();

    let input = Tensor::new(
        ResolvedTensorDims::new(&[4]),
        TensorData::Float(FloatType::F32, vec![1.0, -2.0, 3.5, 0.25]),
    )
    .unwrap();
    let outputs = session.run(std::slice::from_ref(&input)).unwrap();

    assert_eq!(outputs.len(), 1);
    assert!(
        outputs[0].eq_with_epsilon(&input, 1e-6, CompPolicy::Either),
        "world_size=1 all_reduce must be the identity: got {:?}",
        outputs[0]
    );
}
