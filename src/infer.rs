use crate::model::{Graph, Node};
use crate::operator::Operator;
use crate::tensor::{
    resolved_dimensions::broadcast_shape,
    tensor::{ResolvedTensorType, TypeError},
};

impl Graph {
    fn infer_node_output(&self, node: &Node) -> Result<Vec<ResolvedTensorType>, TypeError> {
        let inputs = node
            .inputs
            .iter()
            .map(|&id| {
                self.values[id]
                    .ty
                    .clone()
                    .map(|x| x.to_resolved())
                    .flatten()
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(TypeError::UnresolvedInput)?;

        let mut res: Vec<ResolvedTensorType> = Vec::new();
        match node.op {
            Operator::Add => {
                let a = &inputs[0];
                let b = &inputs[1];
                if a.elem_type != b.elem_type {
                    return Err(TypeError::UnresolvedInput);
                }
                let dims = broadcast_shape(&a.dims, &b.dims)?;
                res.push(ResolvedTensorType {
                    elem_type: a.elem_type.clone(),
                    dims,
                });
            }
        }
        Ok(res)
    }

    pub fn infer(&mut self) -> Result<(), TypeError> {
        for (_, node) in self.nodes.iter() {
            let types = self.infer_node_output(node)?;
            for i in 0..node.outputs.len() {
                let value_id = node.outputs[i];
                self.values[value_id].ty = Some(types[i].clone().into());
            }
        }
        Ok(())
    }
}
