use std::collections::HashSet;

use crate::onnx::operator::args;
use crate::onnx::operator::Operator;
use crate::schedule::*;

pub struct OmpResult(pub HashSet<KernelId>);

pub struct OmpAnnotatePass {
    pub elementwise_threshold: usize,
    pub attention_threshold: usize,
}

impl SchedulePass for OmpAnnotatePass {
    fn summary(&self) -> &str {
        "OpenMP annotation"
    }

    fn run(&self, schedule: &mut Schedule) {
        let result = Annotator {
            elementwise_threshold: self.elementwise_threshold,
            attention_threshold: self.attention_threshold,
        }
        .annotate(schedule);
        schedule.analysis.insert(result);
    }
}

struct Annotator {
    elementwise_threshold: usize,
    attention_threshold: usize,
}

impl Annotator {
    fn annotate(&self, schedule: &Schedule) -> OmpResult {
        let mut set = HashSet::new();

        for (id, kernel) in schedule.kernels.iter() {
            match &kernel.body {
                KernelBody::Opaque(Opaque { op }) => {
                    if let Operator::Attention(_) = op {
                        let q = kernel.inputs[args::ATTENTION_Q];
                        let k = kernel.inputs[args::ATTENTION_K];
                        let q_dims = &schedule.get_resolved_tensor_type(q).unwrap().dims;
                        let k_dims = &schedule.get_resolved_tensor_type(k).unwrap().dims;
                        let num_outer = q_dims[0] * q_dims[1];
                        let seq_q = q_dims[2];
                        let seq_k = k_dims[2];
                        let work = num_outer * seq_q * seq_k;
                        if work < self.attention_threshold {
                            continue;
                        }
                    } else {
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
                    if size < self.elementwise_threshold {
                        continue;
                    }
                }
            };
            set.insert(id);
        }

        OmpResult(set)
    }
}
