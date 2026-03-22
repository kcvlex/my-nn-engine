use std::collections::HashSet;

use crate::onnx::operator::Operator;
use crate::schedule::*;

pub struct OmpResult(pub HashSet<KernelId>);

pub struct OmpAnnotatePass {
    pub elementwise_threshold: usize,
    pub softmax_threshold: usize,
}

impl SchedulePass for OmpAnnotatePass {
    fn summary(&self) -> &str {
        "OpenMP annotation"
    }

    fn run(&self, schedule: &mut Schedule) {
        let result = Annotator {
            elementwise_threshold: self.elementwise_threshold,
            softmax_threshold: self.softmax_threshold,
        }
        .annotate(schedule);
        schedule.analysis.insert(result);
    }
}

struct Annotator {
    elementwise_threshold: usize,
    softmax_threshold: usize,
}

impl Annotator {
    fn annotate(&self, schedule: &Schedule) -> OmpResult {
        let mut set = HashSet::new();

        for (id, kernel) in schedule.kernels.iter() {
            match &kernel.body {
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
                    if size < self.elementwise_threshold {
                        continue;
                    }
                }
                KernelBody::Opaque(Opaque { op }) => {
                    let Operator::Softmax(softmax) = op else {
                        continue;
                    };
                    let input = kernel.inputs[0];
                    let dims = &schedule.get_resolved_tensor_type(input).unwrap().dims;
                    let axis = softmax.axis.index(dims.ndim());
                    if axis != dims.ndim() - 1 {
                        continue;
                    }
                    let outer: usize = dims.iter().take(axis).product();
                    if outer < self.softmax_threshold {
                        continue;
                    }
                }
            };
            set.insert(id);
        }
        OmpResult(set)
    }
}
