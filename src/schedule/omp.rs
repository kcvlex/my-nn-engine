use std::collections::HashSet;

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
        let result = ElementwiseOmp {
            threshold: self.threshold,
        }
        .annotate(schedule);
        schedule.analysis.insert(result);
    }
}

struct ElementwiseOmp {
    threshold: usize,
}

impl ElementwiseOmp {
    fn annotate(&self, schedule: &Schedule) -> OmpResult {
        let mut set = HashSet::new();

        for (id, kernel) in schedule.kernels.iter() {
            let KernelBody::ElementWises(ElementWises { ops }) = &kernel.body else {
                continue;
            };
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
            if self.threshold <= size {
                set.insert(id);
            }
        }

        OmpResult(set)
    }
}
