use crate::onnx::model::{Graph, Node, ValueId};
use crate::onnx::operator::*;
use crate::optimize::optimizer::GraphModifier;
use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use std::io::{Error, Result};

#[derive(Default)]
pub struct TransposeGenerator {
    input: Option<ValueId>,
    perms: Option<Vec<usize>>,
    node_name: Option<String>,
    value_name: Option<String>,
    contiguous: Option<bool>,
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

    pub fn set_contiguous(mut self, contiguous: bool) -> Self {
        self.contiguous = Some(contiguous);
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
        let contiguous = self.contiguous.unwrap_or(false);

        let input_ty = graph.get_resolved_tensor_type(input).unwrap();
        if input_ty.dims.ndim() != perms.len() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "perms length must be equal to input rank",
            ));
        }

        let new_ty = input_ty.transpose(&perms);
        let transposed = modifier.register_new_value(graph, value_name.clone(), new_ty.clone());
        modifier.register_new_node(
            graph,
            Node {
                inputs: vec![input],
                outputs: vec![transposed],
                op: Operator::Transpose(perms),
                name: node_name.clone(),
                mark_as_deleted: false,
            },
        );

        let new_value = if contiguous {
            let new_ty = new_ty.contiguous();
            let new_value =
                modifier.register_new_value(graph, format!("{value_name}_Contiguous"), new_ty);
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![transposed],
                    outputs: vec![new_value],
                    op: Operator::Contiguous,
                    name: format!("{node_name}_Contiguous"),
                    mark_as_deleted: false,
                },
            );
            new_value
        } else {
            transposed
        };
        Ok(new_value)
    }
}

#[derive(Default)]
pub struct ReshapeGenerator {
    input: Option<ValueId>,
    dims: Option<ResolvedTensorDims>,
    node_name: Option<String>,
    value_name: Option<String>,
    allow_contiguous: Option<bool>,
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

    pub fn set_allow_contiguous(mut self, allow_contiguous: bool) -> Self {
        self.allow_contiguous = Some(allow_contiguous);
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
        let allow_contiguous = self.allow_contiguous.unwrap_or(true);
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

        let input_ty = graph.get_resolved_tensor_type(input).unwrap().clone();
        if input_ty.dims.size() != dims.size() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "dims size must be equal to input size",
            ));
        }
        let reshaped_ty = input_ty.try_reshape(&dims);
        let (input_value, reshaped_ty) = match reshaped_ty {
            Some(ty) => (input, ty),
            None => {
                if !allow_contiguous {
                    return Err(Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "cannot reshape because of uncontiguous area",
                    ));
                } else {
                    let new_ty = input_ty.contiguous();
                    let reshaped_ty = new_ty.try_reshape(&dims).unwrap();
                    let value_name = format!("{value_name}_Continguous");
                    let node_name = format!("{node_name}_Continguous");
                    let new_value = modifier.register_new_value(graph, value_name, new_ty.clone());
                    modifier.register_new_node(
                        graph,
                        Node {
                            inputs: vec![input],
                            outputs: vec![new_value],
                            op: Operator::Contiguous,
                            name: node_name,
                            mark_as_deleted: false,
                        },
                    );
                    (new_value, reshaped_ty)
                }
            }
        };

        let new_value = modifier.register_new_value(graph, value_name, reshaped_ty);
        modifier.register_new_node(
            graph,
            Node {
                inputs: vec![input_value],
                outputs: vec![new_value],
                op: Operator::Reshape,
                name: node_name,
                mark_as_deleted: false,
            },
        );
        Ok(new_value)
    }
}
