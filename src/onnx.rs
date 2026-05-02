pub mod load;
pub mod save;

use crate::graph::Graph;

#[derive(Debug, Clone)]
pub struct OpsetImport {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug)]
pub struct Model {
    pub ir_version: i64,
    pub opset_import: Vec<OpsetImport>,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub graph: Graph,
}
