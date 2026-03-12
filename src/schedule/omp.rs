use std::collections::HashMap;

use crate::schedule::*;

#[derive(Debug, Clone, Default)]
pub struct OmpInfo {
    pub omp_for: Option<usize>,
}

pub struct OmpResult(pub HashMap<KernelId, OmpInfo>);

pub struct OmpAnnotatePass {
    pub threshold: usize,
}

impl SchedulePass for OmpAnnotatePass {
    fn summary(&self) -> &str {
        "OpenMP annotation"
    }

    fn run(&self, schedule: &mut Schedule) {
        let result = InnermostOMP {
            threshold: self.threshold,
        }
        .annotate(schedule);
        schedule.analysis.insert(result);
    }
}

struct InnermostOMP {
    threshold: usize,
}

impl InnermostOMP {
    fn annotate(&self, schedule: &Schedule) -> OmpResult {
        let mut map = HashMap::new();

        for (id, kernel) in schedule.kernels.iter() {
            let omp_info = match &kernel.body {
                KernelBody::Opaque(Opaque { op }) => {
                    assert!(!op.is_elementwise());
                    continue;
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
                    let ty = schedule.get_resolved_tensor_type(output).unwrap();
                    let annotate = ty
                        .dims
                        .last()
                        .map(|x| self.threshold <= *x)
                        .unwrap_or(false);
                    if !annotate {
                        continue;
                    }
                    let ndim = ty.dims.ndim();
                    if ty.stride(ndim - 1) != 1 {
                        continue;
                    }
                    let dim = ndim - 1;
                    OmpInfo { omp_for: Some(dim) }
                }
            };

            map.insert(id, omp_info);
        }

        OmpResult(map)
    }
}
