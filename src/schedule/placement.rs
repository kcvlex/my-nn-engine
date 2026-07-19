use std::collections::HashMap;
use std::collections::HashSet;

use itertools::Itertools;

use crate::graph::ValueId;
use crate::schedule::ir::Device;
use crate::schedule::Kernel;
use crate::schedule::KernelId;
use crate::schedule::Schedule;

#[derive(Debug, Clone)]
pub struct Placement {
    devices: HashMap<KernelId, Device>,
}

impl Placement {
    pub fn device_of(&self, kid: KernelId) -> Device {
        self.devices[&kid]
    }

    pub fn iter(&self) -> impl Iterator<Item = (KernelId, Device)> + '_ {
        self.devices.iter().map(|(k, d)| (*k, *d))
    }

    pub fn uniform(schedule: &Schedule, device: Device) -> Self {
        let devices = schedule
            .kernels
            .iter()
            .map(|(kid, _)| (kid, device))
            .collect();
        Self { devices }
    }

    pub fn structural_kv_touch(schedule: &Schedule) -> Self {
        let session_state_set: HashSet<_> = schedule.session_states.iter().copied().collect();
        let devices = schedule
            .kernels
            .iter()
            .map(|(kid, kernel)| {
                let touches_kv = kernel
                    .inputs
                    .iter()
                    .filter_map(|v| v.as_ref())
                    .any(|v| session_state_set.contains(v)) ||
                    kernel.outputs.iter().any(|v| session_state_set.contains(v));
                let device = if touches_kv {
                    Device::CUDA
                } else {
                    Device::CPU
                };
                (kid, device)
            })
            .collect();
        Self { devices }
    }

    /// Partial-offload placement for decode on a model that does not fit in VRAM,
    /// as a single contiguous cut over the kernels in schedule (= build/topo ~=
    /// layer) order: the prefix stays on the GPU until its resident weight reaches
    /// `resident_bytes`, the suffix runs on the CPU (reading its weights -- and KV
    /// cache -- from host RAM). This mirrors llama.cpp's `--n-gpu-layers`.
    ///
    /// The cut never splits a KV cache across devices (the hybrid runtime can't
    /// move a session_state mid-use), so a layer whose KV would straddle the cut
    /// is pushed entirely to the CPU side. Weights < `min_bytes` don't count.
    pub fn resident_budget(schedule: &Schedule, min_bytes: usize, resident_bytes: usize) -> Self {
        let initializers: HashSet<_> = schedule.initializers.iter().copied().collect();
        let session_states: HashSet<_> = schedule.session_states.iter().copied().collect();

        let kernels = schedule.kernels.iter().collect_vec();

        // First/last kernel index at which each KV (session_state) is touched, so
        // the cut can be kept off any KV's live span.
        let mut kv_span: HashMap<ValueId, (usize, usize)> = HashMap::new();
        for (i, (_, kernel)) in kernels.iter().enumerate() {
            for v in kernel.inputs.iter().flatten().chain(kernel.outputs.iter()) {
                if session_states.contains(v) {
                    kv_span.entry(*v).and_modify(|s| s.1 = i).or_insert((i, i));
                }
            }
        }

        // Largest prefix whose resident weight stays within the budget.
        let weight_of = |kernel: &Kernel| -> usize {
            kernel
                .inputs
                .iter()
                .flatten()
                .filter(|v| initializers.contains(v))
                .map(|&v| schedule.value_byte_size(v))
                .filter(|&sz| min_bytes <= sz)
                .sum()
        };
        let mut cut = kernels.len();
        let mut resident = 0usize;
        for (i, (_, kernel)) in kernels.iter().enumerate() {
            let w = weight_of(kernel);
            if resident + w > resident_bytes {
                cut = i;
                break;
            }
            resident += w;
        }

        // Snap the cut off any straddling KV span (its layer goes to the CPU side).
        while 0 < cut && kv_span.values().any(|&(f, l)| f < cut && cut <= l) {
            cut -= 1;
        }

        let devices = kernels
            .iter()
            .enumerate()
            .map(|(i, (kid, _))| {
                let device = if i < cut { Device::CUDA } else { Device::CPU };
                (*kid, device)
            })
            .collect();
        Self { devices }
    }
}
