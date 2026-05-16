use std::collections::HashMap;
use std::collections::HashSet;

use crate::schedule::ir::Device;
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
}
