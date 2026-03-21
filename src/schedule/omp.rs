use std::collections::HashSet;

use crate::onnx::operator::Operator;
use crate::schedule::*;

pub struct OmpResult(pub HashSet<KernelId>);

pub struct OmpAnnotatePass {
    pub threshold: usize,
}

impl SchedulePass for OmpAnnotatePass {
    fn summary(&self) -> &str {
        "OpenMP annotation"
    }

    fn run(&self, schedule: &mut Schedule) {
        let result = Annotator {
            threshold: self.threshold,
        }
        .annotate(schedule);
        schedule.analysis.insert(result);
    }
}

struct Annotator {
    threshold: usize,
}

impl Annotator {
    fn annotate(&self, schedule: &Schedule) -> OmpResult {
        let mut set = HashSet::new();

        for (id, kernel) in schedule.kernels.iter() {
            match &kernel.body {
                KernelBody::Opaque(Opaque { op }) => {
                    if !matches!(op, Operator::Attention(_)) {
                        continue;
                    }
                }

                KernelBody::ElementWises(ElementWises { ops }) => {
                    let skip = ops.iter().any(|(_, args)| {
                        args.iter()
                            .any(|arg| matches!(arg, ElementwiseOpArg::Input(n) if 3 <= *n))
                    });
                    if skip {
                        continue;
                    }

                    let output = kernel.outputs[0];
                    let size = schedule
                        .get_resolved_tensor_type(output)
                        .unwrap()
                        .dims
                        .size();
                    if size < self.threshold {
                        continue;
                    }
                }
            };
            set.insert(id);
        }

        OmpResult(set)
    }
}
