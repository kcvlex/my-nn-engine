use std::collections::HashSet;

use crate::schedule::*;

pub struct OmpResult(pub HashSet<KernelId>);

pub struct OmpAnnotatePass {
    pub elementwise_threshold: usize,
}

impl SchedulePass for OmpAnnotatePass {
    fn summary(&self) -> &str {
        "OpenMP annotation"
    }

    fn run(&self, schedule: &mut Schedule) {
        let result = Annotator {
            elementwise_threshold: self.elementwise_threshold,
        }
        .annotate(schedule);
        schedule.analysis.insert(result);
    }
}

struct Annotator {
    elementwise_threshold: usize,
}

impl Annotator {
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
            if size < self.elementwise_threshold {
                continue;
            }
            set.insert(id);
        }
        OmpResult(set)
    }
}
