use crate::schedule::*;
use crate::tensor::types::DataType;
use indexmap::IndexMap;
use indexmap::map::Entry;

struct MemSize {
    map: IndexMap<DataType, u64>,
}

impl MemSize {
    fn append(&mut self, k: DataType, v: u64) {
        match self.map.entry(k) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = (*entry.get()).max(v);
            }
            Entry::Vacant(entry) => {
                entry.insert(v);
            }
        }
    }
}

pub struct CodeGenContext {
    pub schedule: Schedule,
    mem_sizes: Vec<MemSize>,
}


