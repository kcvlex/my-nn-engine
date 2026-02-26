use crate::schedule::*;

pub struct InnermostOMP {
    pub threshold: usize,
}

impl InnermostOMP {
    pub fn annotate(&self, schedule: &mut Schedule) {
        let ids = schedule
            .kernels
            .iter()
            .filter_map(|(id, kernel)| {
                match &kernel.body {
                    KernelBody::Opaque(Opaque { op }) => {
                        assert!(!op.is_elementwise());
                        return None;
                    }
                    KernelBody::ElementWises(ElementWises { ops }) => {
                        for (_, args) in ops.iter() {
                            for arg in args {
                                match arg {
                                    // TODO: ????
                                    ElementwiseOpArg::Input(n) if 3 <= *n => return None,
                                    _ => (),
                                }
                            }
                        }
                    }
                }

                let output = kernel.outputs[0];
                let ty = &schedule.get_resolved_tensor_type(output).unwrap();
                let annotate = ty
                    .dims
                    .last()
                    .map(|x| self.threshold <= *x)
                    .unwrap_or(false);
                if !annotate {
                    return None;
                }
                let ndim = ty.dims.ndim();
                if ty.stride(ndim - 1) != 1 {
                    return None;
                }
                Some((id, ty.dims.ndim() - 1))
            })
            .collect::<Vec<_>>();

        for (id, dim) in ids.iter() {
            let kernel = &mut schedule.kernels[*id];
            kernel.omp_info.omp_parallel = Some(*dim);
            kernel.omp_info.omp_for = Some(*dim);
        }
    }
}
