use crate::onnx::model::{Graph, NodeId, Nodes, ValueId, ValueInfo};
use crate::onnx::operator::*;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};

pub fn simple_topological_order(graph: &Graph) -> Vec<NodeId> {
    fn dfs(
        node: NodeId,
        res: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
        adj: &HashMap<NodeId, HashSet<NodeId>>,
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
    let mut adj = HashMap::new();
    for (id, node) in graph.nodes.iter() {
        for value in node.inputs.iter() {
            if let Some(defines) = defined.get(value) {
                adj.entry(*defines).or_insert(HashSet::new()).insert(id);
            } else {
                assert!(graph.initializer.contains_key(value));
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InequalityError {
    LeftOnlyIO(String),
    RightOnlyIO(String),
    DifferentComputations(Vec<String>, Vec<String>),
}

pub fn compare_graphs(left: &Graph, right: &Graph) -> Result<(), InequalityError> {
    comp::GraphEquiv::new(left, right).check()
}

mod comp {

    use super::*;
    use indexmap::IndexMap;
    use std::hash::Hash;

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
            let left_init = self.left.graph.initializer.get(&left_id);
            let right_init = self.right.graph.initializer.get(&right_id);
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
                if let Err(err) = self.comp_value(*left_input, *right_input, node_eq) {
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::onnx::load::*;
    use crate::onnx::model::Model;
    use std::path::{Path, PathBuf};

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
}
