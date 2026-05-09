use std::collections::HashMap;
use std::collections::HashSet;

use crate::graph::ValueId;
use crate::schedule::ir::Device;
use crate::schedule::scheduler::value_byte_size;
use crate::schedule::KernelId;
use crate::schedule::Schedule;

/// A kernel can be absorbed into the GPU subgraph if its weight (initializer
/// inputs) is at most this many times its activation traffic
/// (non-initializer inputs + outputs). RMSNorm-class kernels (gain vector
/// much smaller than hidden activation) pass; DequantMatMul / Gemm with
/// weight much larger than activation do not. Coarse proxy for "absorbing
/// this kernel costs less VRAM than the per-step transfer it would force".
const ABSORB_WEIGHT_TO_ACTIVATION_RATIO: usize = 4;

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
}

/// Place every kernel in the attention subgraph on GPU, the rest on CPU. The
/// subgraph grows from kernels that read or write a session_state value
/// (= KV cache update) by absorbing graph neighbors whose weight is at most
/// `ABSORB_WEIGHT_TO_ACTIVATION_RATIO` times their activation traffic. RoPE,
/// QK^T BMM, scale, softmax, score@V BMM, and surrounding RMSNorms get
/// absorbed; weight-heavy projections (Q/K/V/O, FFN, embedding, lm_head) act
/// as natural CPU boundaries.
///
/// Compared to `structural_kv_touch`, this avoids the per-step full-KV-cache
/// D2H transfer: KV cache stays GPU-resident and only small Q (and the
/// attention output) cross the CPU/GPU boundary.
pub fn attention_subgraph(schedule: &Schedule) -> Placement {
    let session_state_set: HashSet<ValueId> = schedule.session_states.iter().copied().collect();
    let initializer_set: HashSet<ValueId> = schedule.initializers.iter().copied().collect();

    let mut value_producer: HashMap<ValueId, KernelId> = HashMap::new();
    let mut value_consumers: HashMap<ValueId, Vec<KernelId>> = HashMap::new();
    for (kid, kernel) in schedule.kernels.iter() {
        for &out in &kernel.outputs {
            value_producer.insert(out, kid);
        }
        for input in kernel.inputs.iter().filter_map(|v| v.as_ref()) {
            value_consumers.entry(*input).or_default().push(kid);
        }
    }

    let weight_bytes = |kid: KernelId| -> usize {
        schedule.kernels[kid]
            .inputs
            .iter()
            .filter_map(|v| v.as_ref())
            .filter(|v| initializer_set.contains(v))
            .map(|v| value_byte_size(schedule, *v))
            .sum()
    };
    // Real data inputs = graph inputs / session_states / values produced by
    // another kernel. Anything else is a workspace (synthetic scratch, e.g.
    // bf16 Gemm dequant buffer) and shouldn't count as activation: workspaces
    // aren't transferred across CPU/GPU and are sized like the weight, so
    // counting them would always pass any weight-vs-activation threshold.
    let graph_inputs: HashSet<ValueId> = schedule.inputs.iter().copied().collect();
    let activation_bytes = |kid: KernelId| -> usize {
        let k = &schedule.kernels[kid];
        let inputs = k
            .inputs
            .iter()
            .filter_map(|v| v.as_ref())
            .filter(|v| {
                !initializer_set.contains(v) &&
                    (graph_inputs.contains(v) ||
                        session_state_set.contains(v) ||
                        value_producer.contains_key(v))
            })
            .map(|v| value_byte_size(schedule, *v))
            .sum::<usize>();
        let outputs = k
            .outputs
            .iter()
            .map(|v| value_byte_size(schedule, *v))
            .sum::<usize>();
        inputs + outputs
    };
    let absorbable = |kid: KernelId| -> bool {
        weight_bytes(kid) <= ABSORB_WEIGHT_TO_ACTIVATION_RATIO * activation_bytes(kid)
    };

    let mut gpu: HashSet<KernelId> = schedule
        .kernels
        .iter()
        .filter_map(|(kid, kernel)| {
            let touches = kernel
                .inputs
                .iter()
                .filter_map(|v| v.as_ref())
                .any(|v| session_state_set.contains(v)) ||
                kernel.outputs.iter().any(|v| session_state_set.contains(v));
            touches.then_some(kid)
        })
        .collect();

    loop {
        let mut grew = false;
        for (kid, kernel) in schedule.kernels.iter() {
            if gpu.contains(&kid) || !absorbable(kid) {
                continue;
            }
            let from_gpu_input = kernel
                .inputs
                .iter()
                .filter_map(|v| v.as_ref())
                .any(|v| value_producer.get(v).is_some_and(|src| gpu.contains(src)));
            let to_gpu_output = kernel.outputs.iter().any(|v| {
                value_consumers
                    .get(v)
                    .map(|cs| cs.iter().any(|c| gpu.contains(c)))
                    .unwrap_or(false)
            });
            if from_gpu_input || to_gpu_output {
                gpu.insert(kid);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    let devices = schedule
        .kernels
        .iter()
        .map(|(kid, _)| {
            let device = if gpu.contains(&kid) {
                Device::CUDA
            } else {
                Device::CPU
            };
            (kid, device)
        })
        .collect();
    Placement { devices }
}

pub fn structural_kv_touch(schedule: &Schedule) -> Placement {
    let session_state_set: HashSet<ValueId> = schedule.session_states.iter().copied().collect();
    let mut devices = HashMap::new();
    for (kid, kernel) in schedule.kernels.iter() {
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
        devices.insert(kid, device);
    }
    Placement { devices }
}
