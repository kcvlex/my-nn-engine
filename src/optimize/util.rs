use crate::model::{Graph, Node, ValueId};
use crate::operator::*;
use crate::optimize::optimizer::GraphModifier;
use crate::tensor::{resolved_dimensions::ResolvedTensorDims, tensor::ResolvedTensorType};
use std::io::{Error, Result};

#[derive(Default)]
pub struct TransposeGenerator {
    input: Option<ValueId>,
    perms: Option<Vec<usize>>,
    node_name: Option<String>,
    value_name: Option<String>,
}

impl TransposeGenerator {
    pub fn set_input(mut self, input: ValueId) -> Self {
        self.input = Some(input);
        self
    }

    pub fn set_perms(mut self, perms: Vec<usize>) -> Self {
        self.perms = Some(perms);
        self
    }

    pub fn set_node_name(mut self, node_name: String) -> Self {
        self.node_name = Some(node_name);
        self
    }

    pub fn set_value_name(mut self, value_name: String) -> Self {
        self.value_name = Some(value_name);
        self
    }

    pub fn generate<T: GraphModifier>(
        self,
        graph: &mut Graph,
        modifier: &mut T,
    ) -> Result<ValueId> {
        let input = self.input.ok_or(Error::new(
            std::io::ErrorKind::InvalidInput,
            "input is required",
        ))?;
        let perms = self.perms.ok_or(Error::new(
            std::io::ErrorKind::InvalidInput,
            "perms is required",
        ))?;
        let node_name = self
            .node_name
            .unwrap_or_else(|| format!("Transpose_{}", input.index()));
        let value_name = self
            .value_name
            .unwrap_or_else(|| format!("Transpose_{}", input.index()));

        let input_ty = graph.get_resolved_tensor_type(input).unwrap();
        if input_ty.dims.ndim() != perms.len() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "perms length must be equal to input rank",
            ));
        }
        let mut output_dim = Vec::with_capacity(perms.len());
        for &perm in perms.iter() {
            output_dim.push(input_ty.dims[perm]);
        }

        let new_value = ResolvedTensorType::new(input_ty.elem_type, output_dim.into());
        let new_value = modifier.register_new_value(graph, value_name, new_value);
        modifier.register_new_node(
            graph,
            Node {
                inputs: vec![input],
                outputs: vec![new_value],
                op: Operator::Transpose(perms),
                name: node_name,
                mark_as_deleted: false,
            },
        );
        Ok(new_value)
    }
}

#[derive(Default)]
pub struct ReshapeGenerator {
    input: Option<ValueId>,
    dims: Option<ResolvedTensorDims>,
    node_name: Option<String>,
    value_name: Option<String>,
}

impl ReshapeGenerator {
    pub fn set_input(mut self, input: ValueId) -> Self {
        self.input = Some(input);
        self
    }

    pub fn set_dims(mut self, dims: ResolvedTensorDims) -> Self {
        self.dims = Some(dims);
        self
    }

    pub fn set_node_name(mut self, node_name: String) -> Self {
        self.node_name = Some(node_name);
        self
    }

    pub fn set_value_name(mut self, value_name: String) -> Self {
        self.value_name = Some(value_name);
        self
    }

    pub fn generate<T: GraphModifier>(
        self,
        graph: &mut Graph,
        modifier: &mut T,
    ) -> Result<ValueId> {
        let input = self.input.ok_or(Error::new(
            std::io::ErrorKind::InvalidInput,
            "input is required",
        ))?;
        let dims = self.dims.ok_or(Error::new(
            std::io::ErrorKind::InvalidInput,
            "dims is required",
        ))?;
        let node_name = self
            .node_name
            .unwrap_or_else(|| format!("Transpose_{}", input.index()));
        let value_name = self
            .value_name
            .unwrap_or_else(|| format!("Transpose_{}", input.index()));

        let input_ty = graph.get_resolved_tensor_type(input).unwrap();
        if input_ty.dims.size() != dims.size() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "dims size must be equal to input size",
            ));
        }

        let new_value = ResolvedTensorType::new(input_ty.elem_type, dims);
        let new_value = modifier.register_new_value(graph, value_name, new_value);
        modifier.register_new_node(
            graph,
            Node {
                inputs: vec![input],
                outputs: vec![new_value],
                op: Operator::Reshape,
                name: node_name,
                mark_as_deleted: false,
            },
        );
        Ok(new_value)
    }
}
