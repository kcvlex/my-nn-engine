use crate::operator::*;

pub trait OptInfo {
    fn is_identity(&self) -> bool;
}

impl OptInfo for Operator {
    fn is_identity(&self) -> bool {
        matches!(
            self,
            Operator::Input(_) | Operator::Output(_) | Operator::Reshape | Operator::ForceReshape
        )
    }
}
