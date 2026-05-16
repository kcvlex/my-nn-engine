use std::collections::HashMap;
use std::collections::HashSet;

use indexmap::IndexMap;
use indexmap::IndexSet;
use itertools::Itertools;

use crate::graph::operator::*;
use crate::graph::Graph;
use crate::graph::NodeId;
use crate::graph::Nodes;
use crate::graph::ValueId;
use crate::graph::ValueInfo;

pub fn simple_topological_order(graph: &Graph) -> Vec<NodeId> {
    fn dfs(
        node: NodeId,
        res: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
        adj: &IndexMap<NodeId, IndexSet<NodeId>>,
        nodes: &Nodes,
    ) {
        if visited.contains(&node) {
            return;
        }
        visited.insert(node);
        if let Some(neighbors) = adj.get(&node) {
            for &next in neighbors.iter() {
                dfs(next, res, visited, adj, nodes);
            }
        }
        if !nodes[node].is_dummy() {
            res.push(node);
        }
    }

    let mut res = Vec::new();
    let mut visited = HashSet::new();
    let mut defined = HashMap::new();
    for (id, node) in graph.nodes.iter() {
        for value in node.outputs.iter() {
            defined.insert(value, id);
        }
    }
    let mut adj = IndexMap::new();
    for (id, node) in graph.nodes.iter() {
        for value in node.inputs.iter().filter_map(|v| v.as_ref()) {
            if let Some(defines) = defined.get(value) {
                adj.entry(*defines).or_insert(IndexSet::new()).insert(id);
            } else {
                assert!(graph.has_initializer(*value));
            }
        }
    }

    for (id, _) in graph.nodes.iter().filter(|(_, node)| !node.is_dummy()) {
        if !visited.contains(&id) {
            dfs(id, &mut res, &mut visited, &adj, &graph.nodes);
        }
    }
    res.reverse();
    res
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InequalityError {
    #[error("left-only IO: {0}")]
    LeftOnlyIO(String),
    #[error("right-only IO: {0}")]
    RightOnlyIO(String),
    #[error("different computations: left={0:?} right={1:?}")]
    DifferentComputations(Vec<String>, Vec<String>),
}

pub fn compare_graphs(left: &Graph, right: &Graph) -> Result<(), InequalityError> {
    comp::GraphEquiv::new(left, right).check()
}

pub fn compare_graphs_structural(left: &Graph, right: &Graph) -> Result<(), InequalityError> {
    comp_structural::GraphStructuralEquiv::new(left, right, None).check()
}

pub fn compare_graphs_structural_with_epsilon(
    left: &Graph,
    right: &Graph,
    epsilon: f64,
) -> Result<(), InequalityError> {
    comp_structural::GraphStructuralEquiv::new(left, right, Some(epsilon)).check()
}

mod comp {

    use std::hash::Hash;

    use indexmap::IndexMap;

    use super::*;

    struct Bijective<T: Eq + Hash + Clone> {
        left2right: HashMap<T, T>,
        right2left: HashMap<T, T>,
    }

    impl<T: Eq + Hash + Clone> Bijective<T> {
        fn new() -> Self {
            Bijective {
                left2right: HashMap::new(),
                right2left: HashMap::new(),
            }
        }

        fn add(&mut self, left: T, right: T) {
            assert!(!self.left2right.contains_key(&left) && !self.right2left.contains_key(&right));
            self.left2right.insert(left.clone(), right.clone());
            self.right2left.insert(right, left);
        }

        fn apply(&self, left: &T) -> Option<&T> {
            self.left2right.get(left)
        }
    }

    struct GraphInfo<'graph> {
        graph: &'graph Graph,
        inputs: IndexMap<String, ValueId>,
        outputs: IndexMap<String, ValueId>,
        defined: HashMap<ValueId, (NodeId, usize)>,
    }

    impl<'graph> GraphInfo<'graph> {
        fn new(graph: &'graph Graph) -> Self {
            let inputs = graph
                .inputs
                .iter()
                .map(|n| match graph.nodes[*n].op {
                    Operator::Input(v) => (graph.values[v].name.clone(), v),
                    _ => panic!("Expected input operator"),
                })
                .collect();
            let outputs = graph
                .outputs
                .iter()
                .map(|n| match graph.nodes[*n].op {
                    Operator::Output(v) => (graph.values[v].name.clone(), v),
                    _ => panic!("Expected output operator"),
                })
                .collect();
            let defined = graph
                .nodes
                .iter()
                .filter(|(_, node)| !node.is_dummy())
                .flat_map(|(id, node)| {
                    node.outputs
                        .iter()
                        .enumerate()
                        .map(move |(i, v)| (*v, (id, i)))
                })
                .collect();
            GraphInfo {
                graph,
                inputs,
                outputs,
                defined,
            }
        }
    }

    pub struct GraphEquiv<'graph> {
        left: GraphInfo<'graph>,
        right: GraphInfo<'graph>,
    }

    impl<'graph> GraphEquiv<'graph> {
        pub fn new(left: &'graph Graph, right: &'graph Graph) -> Self {
            let left = GraphInfo::new(left);
            let right = GraphInfo::new(right);
            GraphEquiv { left, right }
        }

        fn comp_valueinfo(
            &self,
            left: &ValueInfo,
            right: &ValueInfo,
        ) -> Result<(), InequalityError> {
            if *left != *right {
                return Err(InequalityError::DifferentComputations(
                    vec![left.name.clone()],
                    vec![right.name.clone()],
                ));
            }
            Ok(())
        }

        fn comp_io<const IS_INPUT: bool>(&self) -> Result<(), InequalityError> {
            let lefts = if IS_INPUT {
                &self.left.inputs
            } else {
                &self.left.outputs
            };
            let rights = if IS_INPUT {
                &self.right.inputs
            } else {
                &self.right.outputs
            };

            for (name, left) in lefts.iter() {
                let left = &self.left.graph.values[*left];
                let right = match rights.get(name) {
                    Some(right) => &self.right.graph.values[*right],
                    None => return Err(InequalityError::LeftOnlyIO(name.clone())),
                };
                self.comp_valueinfo(left, right)?;
            }

            if lefts.len() != rights.len() {
                for (name, _) in rights.iter() {
                    if !lefts.contains_key(name) {
                        return Err(InequalityError::RightOnlyIO(name.clone()));
                    }
                }
                unreachable!();
            }

            Ok(())
        }

        fn comp_value(
            &self,
            left_id: ValueId,
            right_id: ValueId,
            node_eq: &mut Bijective<NodeId>,
        ) -> Result<(), InequalityError> {
            let left = &self.left.graph.values[left_id];
            let right = &self.right.graph.values[right_id];
            let left_init = self.left.graph.initializer.get(&left_id).cloned();
            let right_init = self.right.graph.initializer.get(&right_id).cloned();
            match (left_init, right_init) {
                (Some(lv), Some(rv)) if lv == rv => {
                    return Ok(());
                }
                (Some(_), None) | (None, Some(_)) | (Some(_), Some(_)) => {
                    return Err(InequalityError::DifferentComputations(
                        vec![left.name.clone()],
                        vec![right.name.clone()],
                    ));
                }
                (None, None) => (),
            }

            if left.name == right.name &&
                self.left.inputs.contains_key(&left.name) &&
                self.right.inputs.contains_key(&right.name)
            {
                return Ok(());
            }

            let (left_node_id, left_idx) = *self.left.defined.get(&left_id).unwrap();
            let (right_node_id, right_idx) = *self.right.defined.get(&right_id).unwrap();
            let left_node = &self.left.graph.nodes[left_node_id];
            let right_node = &self.right.graph.nodes[right_node_id];

            let make_err = || {
                Err(InequalityError::DifferentComputations(
                    vec![left_node.name.clone()],
                    vec![right_node.name.clone()],
                ))
            };

            if left_idx != right_idx {
                return make_err();
            }

            if let Some(correspond) = node_eq.apply(&left_node_id) {
                if *correspond == right_node_id {
                    return Ok(());
                } else {
                    return make_err();
                }
            }

            if left_node.op != right_node.op {
                return make_err();
            }

            for (left_input, right_input) in
                left_node.inputs.iter().zip_eq(right_node.inputs.iter())
            {
                match (left_input, right_input) {
                    (Some(li), Some(ri)) => {
                        if let Err(err) = self.comp_value(*li, *ri, node_eq) {
                            let err = match err {
                                InequalityError::DifferentComputations(mut err0, mut err1) => {
                                    err0.push(left_node.name.clone());
                                    err1.push(right_node.name.clone());
                                    InequalityError::DifferentComputations(err0, err1)
                                }
                                _ => unreachable!(),
                            };
                            return Err(err);
                        }
                    }
                    (None, None) => {}
                    _ => return make_err(),
                }
            }

            node_eq.add(left_node_id, right_node_id);
            Ok(())
        }

        pub fn check(&mut self) -> Result<(), InequalityError> {
            self.comp_io::<true>()?;
            self.comp_io::<false>()?;

            assert!(self.left.outputs.len() == self.right.outputs.len());

            let mut node_eq = Bijective::new();
            for (name, left_id) in self.left.outputs.iter() {
                let right_id = self.right.outputs.get(name).unwrap();
                self.comp_value(*left_id, *right_id, &mut node_eq)?;
            }
            Ok(())
        }
    }
}

mod comp_structural {
    use std::hash::Hash;

    use super::*;

    struct Bijective<T: Eq + Hash + Clone> {
        left2right: HashMap<T, T>,
        right2left: HashMap<T, T>,
    }

    impl<T: Eq + Hash + Clone> Bijective<T> {
        fn new() -> Self {
            Bijective {
                left2right: HashMap::new(),
                right2left: HashMap::new(),
            }
        }

        fn add(&mut self, left: T, right: T) -> Result<(), ()> {
            if let Some(existing) = self.left2right.get(&left) {
                return if *existing == right { Ok(()) } else { Err(()) };
            }
            if self.right2left.contains_key(&right) {
                return Err(());
            }
            self.left2right.insert(left.clone(), right.clone());
            self.right2left.insert(right, left);
            Ok(())
        }

        fn get(&self, left: &T) -> Option<&T> {
            self.left2right.get(left)
        }
    }

    struct GraphInfo<'graph> {
        graph: &'graph Graph,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        defined: HashMap<ValueId, (NodeId, usize)>,
    }

    impl<'graph> GraphInfo<'graph> {
        fn new(graph: &'graph Graph) -> Self {
            let inputs = graph
                .inputs
                .iter()
                .map(|n| match graph.nodes[*n].op {
                    Operator::Input(v) => v,
                    _ => panic!("Expected input operator"),
                })
                .collect();
            let outputs = graph
                .outputs
                .iter()
                .map(|n| match graph.nodes[*n].op {
                    Operator::Output(v) => v,
                    _ => panic!("Expected output operator"),
                })
                .collect();
            let defined = graph
                .nodes
                .iter()
                .filter(|(_, node)| !node.is_dummy())
                .flat_map(|(id, node)| {
                    node.outputs
                        .iter()
                        .enumerate()
                        .map(move |(i, v)| (*v, (id, i)))
                })
                .collect();
            GraphInfo {
                graph,
                inputs,
                outputs,
                defined,
            }
        }
    }

    pub struct GraphStructuralEquiv<'graph> {
        left: GraphInfo<'graph>,
        right: GraphInfo<'graph>,
        epsilon: Option<f64>,
    }

    impl<'graph> GraphStructuralEquiv<'graph> {
        pub fn new(left: &'graph Graph, right: &'graph Graph, epsilon: Option<f64>) -> Self {
            GraphStructuralEquiv {
                left: GraphInfo::new(left),
                right: GraphInfo::new(right),
                epsilon,
            }
        }

        fn make_err(left_node: &str, right_node: &str) -> InequalityError {
            InequalityError::DifferentComputations(
                vec![left_node.to_string()],
                vec![right_node.to_string()],
            )
        }

        fn comp_value(
            &self,
            left_id: ValueId,
            right_id: ValueId,
            value_eq: &mut Bijective<ValueId>,
            node_eq: &mut Bijective<NodeId>,
        ) -> Result<(), InequalityError> {
            if value_eq.get(&left_id) == Some(&right_id) {
                return Ok(());
            }

            // Both initializers
            let left_init = self.left.graph.initializer.get(&left_id).cloned();
            let right_init = self.right.graph.initializer.get(&right_id).cloned();
            let inits_equal = match (&left_init, &right_init) {
                (Some(lv), Some(rv)) => match self.epsilon {
                    Some(eps) => {
                        lv.eq_with_epsilon(rv, eps, crate::tensor::data::CompPolicy::Either)
                    }
                    None => lv == rv,
                },
                _ => false,
            };
            match (left_init, right_init) {
                (Some(_), Some(_)) if inits_equal => {
                    value_eq.add(left_id, right_id).map_err(|_| {
                        Self::make_err(
                            &self.left.graph.values[left_id].name,
                            &self.right.graph.values[right_id].name,
                        )
                    })?;
                    return Ok(());
                }
                (Some(_), None) | (None, Some(_)) | (Some(_), Some(_)) => {
                    return Err(Self::make_err(
                        &self.left.graph.values[left_id].name,
                        &self.right.graph.values[right_id].name,
                    ));
                }
                (None, None) => (),
            }

            // Both graph inputs (matched by position)
            let left_input_pos = self.left.inputs.iter().position(|v| *v == left_id);
            let right_input_pos = self.right.inputs.iter().position(|v| *v == right_id);
            match (left_input_pos, right_input_pos) {
                (Some(l), Some(r)) if l == r => {
                    value_eq.add(left_id, right_id).map_err(|_| {
                        Self::make_err(
                            &self.left.graph.values[left_id].name,
                            &self.right.graph.values[right_id].name,
                        )
                    })?;
                    return Ok(());
                }
                (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) => {
                    return Err(Self::make_err(
                        &self.left.graph.values[left_id].name,
                        &self.right.graph.values[right_id].name,
                    ));
                }
                (None, None) => (),
            }

            // Defined by nodes - compare defining nodes
            let (left_node_id, left_idx) = *self.left.defined.get(&left_id).unwrap();
            let (right_node_id, right_idx) = *self.right.defined.get(&right_id).unwrap();
            let left_node = &self.left.graph.nodes[left_node_id];
            let right_node = &self.right.graph.nodes[right_node_id];

            if left_idx != right_idx {
                return Err(Self::make_err(&left_node.name, &right_node.name));
            }

            if let Some(correspond) = node_eq.get(&left_node_id) {
                if *correspond == right_node_id {
                    value_eq
                        .add(left_id, right_id)
                        .map_err(|_| Self::make_err(&left_node.name, &right_node.name))?;
                    return Ok(());
                } else {
                    return Err(Self::make_err(&left_node.name, &right_node.name));
                }
            }

            if left_node.op != right_node.op {
                return Err(Self::make_err(&left_node.name, &right_node.name));
            }

            if left_node.inputs.len() != right_node.inputs.len() {
                return Err(Self::make_err(&left_node.name, &right_node.name));
            }

            for (left_input, right_input) in left_node.inputs.iter().zip(right_node.inputs.iter()) {
                match (left_input, right_input) {
                    (Some(li), Some(ri)) => {
                        self.comp_value(*li, *ri, value_eq, node_eq)?;
                    }
                    (None, None) => {}
                    _ => {
                        return Err(Self::make_err(&left_node.name, &right_node.name));
                    }
                }
            }

            node_eq
                .add(left_node_id, right_node_id)
                .map_err(|_| Self::make_err(&left_node.name, &right_node.name))?;
            value_eq
                .add(left_id, right_id)
                .map_err(|_| Self::make_err(&left_node.name, &right_node.name))?;
            Ok(())
        }

        pub fn check(&self) -> Result<(), InequalityError> {
            if self.left.inputs.len() != self.right.inputs.len() {
                return Err(InequalityError::DifferentComputations(
                    vec!["input count mismatch".to_string()],
                    vec![],
                ));
            }
            if self.left.outputs.len() != self.right.outputs.len() {
                return Err(InequalityError::DifferentComputations(
                    vec!["output count mismatch".to_string()],
                    vec![],
                ));
            }

            // Check input/output types match
            for (l, r) in self.left.inputs.iter().zip(self.right.inputs.iter()) {
                let lt = &self.left.graph.values[*l].ty;
                let rt = &self.right.graph.values[*r].ty;
                if lt != rt {
                    return Err(Self::make_err(
                        &self.left.graph.values[*l].name,
                        &self.right.graph.values[*r].name,
                    ));
                }
            }
            for (l, r) in self.left.outputs.iter().zip(self.right.outputs.iter()) {
                let lt = &self.left.graph.values[*l].ty;
                let rt = &self.right.graph.values[*r].ty;
                if lt != rt {
                    return Err(Self::make_err(
                        &self.left.graph.values[*l].name,
                        &self.right.graph.values[*r].name,
                    ));
                }
            }

            let mut value_eq = Bijective::new();
            let mut node_eq = Bijective::new();
            for (left_id, right_id) in self.left.outputs.iter().zip(self.right.outputs.iter()) {
                self.comp_value(*left_id, *right_id, &mut value_eq, &mut node_eq)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod test {
    use std::path::Path;
    use std::path::PathBuf;

    use super::*;
    use crate::onnx::load::*;
    use crate::onnx::Model;

    fn compare_models<P0: AsRef<Path>, P1: AsRef<Path>>(
        p0: P0,
        p1: P1,
    ) -> Result<(), InequalityError> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/test/onnx");
        let p0 = dir.join(p0);
        let p1 = dir.join(p1);
        let model0 = Model::load_from_path(p0).expect("failed to load");
        let model1 = Model::load_from_path(p1).expect("failed to load");
        compare_graphs(&model0.graph, &model1.graph)
    }

    #[test]
    fn test_mnist_mnist() {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/validated/mnist-12/mnist-12.onnx");
        let model = Model::load_from_path(model).expect("failed to load");
        assert!(compare_graphs(&model.graph, &model.graph).is_ok());
    }

    #[test]
    fn test_equiv() {
        assert_eq!(
            compare_models("add_add_sub0.onnx", "add_add_sub1.onnx"),
            Ok(())
        )
    }

    #[test]
    fn test_noncomutative() {
        assert_eq!(
            compare_models("add_add_sub0.onnx", "add_add_sub2.onnx"),
            Err(InequalityError::DifferentComputations(
                vec![
                    "/layer0/Sigmoid".to_string(),
                    "/Sub".to_string(),
                    "/layer1/Sigmoid".to_string(),
                ],
                vec![
                    "/Add".to_string(),
                    "/Sub".to_string(),
                    "/layer0/Sigmoid".to_string(),
                ],
            ),)
        )
    }

    #[test]
    fn test_same_const() {
        assert!(compare_models("add_const0.onnx", "add_const0.onnx").is_ok());
    }

    #[test]
    fn test_different_const() {
        let err = compare_models("add_const0.onnx", "add_const1.onnx");
        assert!(err.is_err());
    }

    use crate::graph::Node;
    use crate::graph::ValueInfo;
    use crate::tensor::types::FloatType;
    use crate::tensor::types::ResolvedTensorDims;
    use crate::tensor::types::ResolvedTensorType;
    use crate::tensor::types::TensorType;

    fn make_value(graph: &mut Graph, name: &str, dims: &[usize]) -> ValueId {
        graph.values.alloc(ValueInfo {
            name: name.to_string(),
            ty: Some(TensorType::Resolved(ResolvedTensorType::new(
                FloatType::F32.into(),
                ResolvedTensorDims::new(dims),
            ))),
        })
    }

    fn make_graph(name_prefix: &str) -> Graph {
        // x -> Sigmoid -> Add(a,a) -> y
        let mut graph = Graph::empty_graph("test".to_string());
        let x = make_value(&mut graph, &format!("{name_prefix}_x"), &[2, 3]);
        let a = make_value(&mut graph, &format!("{name_prefix}_a"), &[2, 3]);
        let y = make_value(&mut graph, &format!("{name_prefix}_y"), &[2, 3]);

        let input_node = graph.nodes.alloc(Node::create_node(
            vec![],
            vec![x],
            format!("{name_prefix}_Input"),
            Operator::Input(x),
        ));
        graph.inputs.push(input_node);

        graph.nodes.alloc(Node::create_node(
            vec![Some(x)],
            vec![a],
            format!("{name_prefix}_Sigmoid"),
            Operator::Sigmoid,
        ));

        graph.nodes.alloc(Node::create_node(
            vec![Some(a), Some(a)],
            vec![y],
            format!("{name_prefix}_Add"),
            Operator::Add,
        ));

        let output_node = graph.nodes.alloc(Node::create_node(
            vec![Some(y)],
            vec![],
            format!("{name_prefix}_Output"),
            Operator::Output(y),
        ));
        graph.outputs.push(output_node);

        graph
    }

    #[test]
    fn test_structural_same_graph() {
        let g1 = make_graph("left");
        let g2 = make_graph("right");
        assert!(compare_graphs_structural(&g1, &g2).is_ok());
    }

    #[test]
    fn test_structural_self() {
        let g = make_graph("g");
        assert!(compare_graphs_structural(&g, &g).is_ok());
    }

    #[test]
    fn test_structural_different_op() {
        // x -> Sigmoid -> Add(a,a) -> y  vs  x -> Tanh -> Add(a,a) -> y
        let g1 = make_graph("left");

        let mut g2 = Graph::empty_graph("test".to_string());
        let x = make_value(&mut g2, "x", &[2, 3]);
        let a = make_value(&mut g2, "a", &[2, 3]);
        let y = make_value(&mut g2, "y", &[2, 3]);
        let input_node = g2.nodes.alloc(Node::create_node(
            vec![],
            vec![x],
            "Input".to_string(),
            Operator::Input(x),
        ));
        g2.inputs.push(input_node);
        g2.nodes.alloc(Node::create_node(
            vec![Some(x)],
            vec![a],
            "Tanh".to_string(),
            Operator::Tanh,
        ));
        g2.nodes.alloc(Node::create_node(
            vec![Some(a), Some(a)],
            vec![y],
            "Add".to_string(),
            Operator::Add,
        ));
        let output_node = g2.nodes.alloc(Node::create_node(
            vec![Some(y)],
            vec![],
            "Output".to_string(),
            Operator::Output(y),
        ));
        g2.outputs.push(output_node);

        assert!(compare_graphs_structural(&g1, &g2).is_err());
    }

    #[test]
    fn test_structural_different_type() {
        // Same structure but different dims
        let g1 = make_graph("left");

        let mut g2 = Graph::empty_graph("test".to_string());
        let x = make_value(&mut g2, "x", &[4, 5]); // different dims
        let a = make_value(&mut g2, "a", &[4, 5]);
        let y = make_value(&mut g2, "y", &[4, 5]);
        let input_node = g2.nodes.alloc(Node::create_node(
            vec![],
            vec![x],
            "Input".to_string(),
            Operator::Input(x),
        ));
        g2.inputs.push(input_node);
        g2.nodes.alloc(Node::create_node(
            vec![Some(x)],
            vec![a],
            "Sigmoid".to_string(),
            Operator::Sigmoid,
        ));
        g2.nodes.alloc(Node::create_node(
            vec![Some(a), Some(a)],
            vec![y],
            "Add".to_string(),
            Operator::Add,
        ));
        let output_node = g2.nodes.alloc(Node::create_node(
            vec![Some(y)],
            vec![],
            "Output".to_string(),
            Operator::Output(y),
        ));
        g2.outputs.push(output_node);

        assert!(compare_graphs_structural(&g1, &g2).is_err());
    }

    #[test]
    fn test_structural_different_connectivity() {
        // x -> Sigmoid -> Add(a,a) -> y  vs  x -> Sigmoid -> Add(x,a) -> y
        let g1 = make_graph("left");

        let mut g2 = Graph::empty_graph("test".to_string());
        let x = make_value(&mut g2, "x", &[2, 3]);
        let a = make_value(&mut g2, "a", &[2, 3]);
        let y = make_value(&mut g2, "y", &[2, 3]);
        let input_node = g2.nodes.alloc(Node::create_node(
            vec![],
            vec![x],
            "Input".to_string(),
            Operator::Input(x),
        ));
        g2.inputs.push(input_node);
        g2.nodes.alloc(Node::create_node(
            vec![Some(x)],
            vec![a],
            "Sigmoid".to_string(),
            Operator::Sigmoid,
        ));
        g2.nodes.alloc(Node::create_node(
            vec![Some(x), Some(a)],
            vec![y], // Add(x,a) instead of Add(a,a)
            "Add".to_string(),
            Operator::Add,
        ));
        let output_node = g2.nodes.alloc(Node::create_node(
            vec![Some(y)],
            vec![],
            "Output".to_string(),
            Operator::Output(y),
        ));
        g2.outputs.push(output_node);

        assert!(compare_graphs_structural(&g1, &g2).is_err());
    }
}
