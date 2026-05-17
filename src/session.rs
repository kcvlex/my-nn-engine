mod cpu;
mod cuda;
mod device_buffer;
mod hybrid;
mod shared_lib;

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

pub use cuda::cuda_lock;
pub use device_buffer::CudaError;
pub use device_buffer::DeviceBuffer;
use log::info;
use tempfile::TempDir;

use crate::codegen::CodeGenError;
use crate::graph::Graph;
use crate::graph::ValueId;
use crate::onnx::load::*;
use crate::onnx::Model;
use crate::options::*;
use crate::schedule::create_schedule_passes;
use crate::schedule::Schedule;
use crate::session::cpu::SessionCPU;
use crate::session::cuda::SessionCUDA;
use crate::session::hybrid::SessionHybrid;
use crate::tensor::data::TensorData;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::TypeError;
use crate::tensor::types::UIntType;
use crate::tensor::Tensor;
use crate::transform::transform_graph;

pub(crate) enum StrictTensor {
    U8(Vec<u8>),
    I8(Vec<i8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    BF16(Vec<u16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

macro_rules! cast_vec {
    ($data: expr, $ty: ty) => {{
        $data.iter().map(|x| *x as $ty).collect::<Vec<_>>()
    }};
}

impl StrictTensor {
    fn zeros(ty: DataType, dims: &ResolvedTensorDims) -> Self {
        let sz = dims.size().max(1);
        match ty {
            DataType::Bool | DataType::UInt(UIntType::U8) => StrictTensor::U8(vec![0; sz]),
            DataType::SInt(SIntType::I8) => StrictTensor::I8(vec![0; sz]),
            DataType::SInt(SIntType::I32) => StrictTensor::I32(vec![0; sz]),
            DataType::SInt(SIntType::I64) => StrictTensor::I64(vec![0; sz]),
            DataType::UInt(UIntType::U64) => StrictTensor::U64(vec![0; sz]),
            DataType::Float(FloatType::BF16) => StrictTensor::BF16(vec![0; sz]),
            DataType::Float(FloatType::F32) => StrictTensor::F32(vec![0.0; sz]),
            DataType::Float(FloatType::F64) => StrictTensor::F64(vec![0.0; sz]),
        }
    }

    #[allow(clippy::unnecessary_cast)]
    fn as_ptr(&self) -> *const u8 {
        match self {
            StrictTensor::U8(v) => v.as_ptr() as *const u8,
            StrictTensor::I8(v) => v.as_ptr() as *const u8,
            StrictTensor::I32(v) => v.as_ptr() as *const u8,
            StrictTensor::I64(v) => v.as_ptr() as *const u8,
            StrictTensor::U64(v) => v.as_ptr() as *const u8,
            StrictTensor::BF16(v) => v.as_ptr() as *const u8,
            StrictTensor::F32(v) => v.as_ptr() as *const u8,
            StrictTensor::F64(v) => v.as_ptr() as *const u8,
        }
    }

    #[allow(clippy::unnecessary_cast)]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            StrictTensor::U8(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::I8(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::I32(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::I64(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::U64(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::BF16(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::F32(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::F64(v) => v.as_mut_ptr() as *mut u8,
        }
    }

    fn byte_len(&self) -> usize {
        match self {
            StrictTensor::U8(v) => v.len(),
            StrictTensor::I8(v) => v.len(),
            StrictTensor::I32(v) => v.len() * std::mem::size_of::<i32>(),
            StrictTensor::I64(v) => v.len() * std::mem::size_of::<i64>(),
            StrictTensor::U64(v) => v.len() * std::mem::size_of::<u64>(),
            StrictTensor::BF16(v) => v.len() * std::mem::size_of::<u16>(),
            StrictTensor::F32(v) => v.len() * std::mem::size_of::<f32>(),
            StrictTensor::F64(v) => v.len() * std::mem::size_of::<f64>(),
        }
    }

    fn into_tensor(self, dims: ResolvedTensorDims) -> Tensor {
        match self {
            StrictTensor::U8(v) => Tensor::new(
                dims,
                TensorData::Bool(v.iter().map(|&b| if b != 0 { 1u8 } else { 0u8 }).collect()),
            )
            .unwrap(),
            StrictTensor::I8(v) => {
                Tensor::new(dims, TensorData::SInt(SIntType::I8, cast_vec!(v, i64))).unwrap()
            }
            StrictTensor::I32(v) => {
                Tensor::new(dims, TensorData::SInt(SIntType::I32, cast_vec!(v, i64))).unwrap()
            }
            StrictTensor::I64(v) => Tensor::new(dims, TensorData::SInt(SIntType::I64, v)).unwrap(),
            StrictTensor::U64(v) => Tensor::new(dims, TensorData::UInt(UIntType::U64, v)).unwrap(),
            StrictTensor::F32(v) => {
                Tensor::new(dims, TensorData::Float(FloatType::F32, cast_vec!(v, f64))).unwrap()
            }
            StrictTensor::F64(v) => {
                Tensor::new(dims, TensorData::Float(FloatType::F64, v)).unwrap()
            }
            StrictTensor::BF16(v) => {
                let f64_vec: Vec<f64> = v
                    .iter()
                    .map(|&bits| f32::from_bits((bits as u32) << 16) as f64)
                    .collect();
                Tensor::new(dims, TensorData::Float(FloatType::BF16, f64_vec)).unwrap()
            }
        }
    }
}

impl From<&Tensor> for StrictTensor {
    fn from(t: &Tensor) -> Self {
        match &t.data {
            TensorData::Bool(v) => StrictTensor::U8(v.clone()),
            TensorData::UInt(UIntType::U8, v) => StrictTensor::U8(cast_vec!(v, u8)),
            TensorData::SInt(SIntType::I8, v) => StrictTensor::I8(cast_vec!(v, i8)),
            TensorData::SInt(SIntType::I32, v) => StrictTensor::I32(cast_vec!(v, i32)),
            TensorData::SInt(SIntType::I64, v) => StrictTensor::I64(v.clone()),
            TensorData::UInt(UIntType::U64, v) => StrictTensor::U64(v.clone()),
            TensorData::Float(FloatType::F32, v) => StrictTensor::F32(cast_vec!(v, f32)),
            TensorData::Float(FloatType::F64, v) => StrictTensor::F64(v.clone()),
            TensorData::Float(FloatType::BF16, v) => StrictTensor::BF16(
                v.iter()
                    .map(|&x| ((x as f32).to_bits() >> 16) as u16)
                    .collect(),
            ),
        }
    }
}

impl StrictTensor {
    fn from_bytes(ty: DataType, raw: &[u8]) -> Self {
        macro_rules! parse {
            ($ctor: expr, $t: ty) => {{
                let sz = std::mem::size_of::<$t>();
                let vec: Vec<$t> = raw
                    .chunks_exact(sz)
                    .map(|b| <$t>::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                $ctor(vec)
            }};
        }
        match ty {
            DataType::Bool | DataType::UInt(UIntType::U8) => StrictTensor::U8(raw.to_vec()),
            DataType::SInt(SIntType::I8) => parse!(StrictTensor::I8, i8),
            DataType::SInt(SIntType::I32) => parse!(StrictTensor::I32, i32),
            DataType::SInt(SIntType::I64) => parse!(StrictTensor::I64, i64),
            DataType::UInt(UIntType::U64) => parse!(StrictTensor::U64, u64),
            DataType::Float(FloatType::BF16) => parse!(StrictTensor::BF16, u16),
            DataType::Float(FloatType::F32) => parse!(StrictTensor::F32, f32),
            DataType::Float(FloatType::F64) => parse!(StrictTensor::F64, f64),
        }
    }
}

pub(crate) enum InitializerSource {
    Inline(StrictTensor),
    External {
        file: Arc<File>,
        offset: u64,
        length: u64,
        elem_type: DataType,
    },
}

impl InitializerSource {
    pub(crate) fn byte_len(&self) -> usize {
        match self {
            InitializerSource::Inline(t) => t.byte_len(),
            InitializerSource::External { length, .. } => *length as usize,
        }
    }

    pub(crate) fn load_into_strict(&self) -> Result<StrictTensor, ModelLoadError> {
        match self {
            InitializerSource::Inline(t) => Ok(t.clone_strict()),
            InitializerSource::External {
                file,
                offset,
                length,
                elem_type,
            } => {
                use std::os::unix::fs::FileExt;
                let mut buf = vec![0u8; *length as usize];
                file.read_exact_at(&mut buf, *offset)
                    .map_err(ModelLoadError::FileRead)?;
                Ok(StrictTensor::from_bytes(*elem_type, &buf))
            }
        }
    }
}

impl StrictTensor {
    fn clone_strict(&self) -> Self {
        match self {
            StrictTensor::U8(v) => StrictTensor::U8(v.clone()),
            StrictTensor::I8(v) => StrictTensor::I8(v.clone()),
            StrictTensor::I32(v) => StrictTensor::I32(v.clone()),
            StrictTensor::I64(v) => StrictTensor::I64(v.clone()),
            StrictTensor::U64(v) => StrictTensor::U64(v.clone()),
            StrictTensor::BF16(v) => StrictTensor::BF16(v.clone()),
            StrictTensor::F32(v) => StrictTensor::F32(v.clone()),
            StrictTensor::F64(v) => StrictTensor::F64(v.clone()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("codegen error: {0}")]
    CodeGenError(#[from] CodeGenError),
    #[error("model load error: {0}")]
    ModelLoadError(#[from] ModelLoadError),
    #[error("type error: {0}")]
    TypeError(#[from] TypeError),
    #[error("{0}")]
    OtherError(String),
}

unsafe impl Send for SessionError {}

#[allow(clippy::large_enum_variant)]
pub enum SessionInner {
    CPU(SessionCPU),
    CUDA(SessionCUDA),
    Hybrid(SessionHybrid),
}

pub struct Session {
    inner: SessionInner,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SessionStateSpec {
    pub name: String,
    pub buffer: Arc<DeviceBuffer>,
}

#[derive(Debug, Default)]
pub struct InitializerBuffers {
    buffers: Mutex<HashMap<String, Arc<DeviceBuffer>>>,
}

impl InitializerBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get_or_insert_with<F>(
        &self,
        name: &str,
        f: F,
    ) -> Result<Arc<DeviceBuffer>, SessionError>
    where
        F: FnOnce() -> Result<Arc<DeviceBuffer>, SessionError>,
    {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(buf) = buffers.get(name) {
            return Ok(Arc::clone(buf));
        }
        let buf = f()?;
        buffers.insert(name.to_string(), Arc::clone(&buf));
        Ok(buf)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    pub session_states: Vec<SessionStateSpec>,
    pub initializer_buffers: Option<Arc<InitializerBuffers>>,
}

pub(crate) fn send_initializer_to_device(
    src: &InitializerSource,
    buf: &DeviceBuffer,
) -> Result<(), SessionError> {
    let len = src.byte_len();
    if len == 0 {
        return Ok(());
    }
    match src {
        InitializerSource::Inline(t) => unsafe {
            buf.host_to_device(t.as_ptr() as *const _, len)
                .map_err(|e| SessionError::OtherError(format!("cudaMemcpy: {:?}", e)))
        },
        InitializerSource::External { file, offset, .. } => {
            let mut tmp = vec![0u8; len];
            file.read_exact_at(&mut tmp, *offset)
                .map_err(ModelLoadError::FileRead)
                .map_err(SessionError::ModelLoadError)?;
            unsafe {
                buf.host_to_device(tmp.as_ptr() as *const _, len)
                    .map_err(|e| SessionError::OtherError(format!("cudaMemcpy: {:?}", e)))
            }
        }
    }
}

fn get_argument_types(
    graph: &Graph,
    values: &[ValueId],
) -> Result<Vec<ResolvedTensorType>, SessionError> {
    values
        .iter()
        .map(|&id| graph.get_resolved_tensor_type(id).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or(SessionError::TypeError(TypeError::UnresolvedInput))
}

impl Session {
    pub fn new<P: AsRef<Path>>(
        p: P,
        input_ty: Option<&[ResolvedTensorType]>,
        options: &Options,
        config: &SessionConfig,
    ) -> Result<Self, SessionError> {
        let _ = env_logger::try_init();
        info!("Session starting");

        let model = Model::load_from_path(p).map_err(SessionError::ModelLoadError)?;
        info!("Model loaded");

        Self::from_graph_inner(model.graph, input_ty, options, config)
    }

    pub fn from_graph(
        graph: Graph,
        options: &Options,
        config: &SessionConfig,
    ) -> Result<Self, SessionError> {
        let _ = env_logger::try_init();
        Self::from_graph_inner(graph, None, options, config)
    }

    /// Load a model and resolve unknown input shapes from the actual
    /// `inputs` map (matched by `TensorProto.name`). Useful for models
    /// declaring symbolic dimensions like `N` for batch size.
    pub fn new_with_inputs<P: AsRef<Path>>(
        p: P,
        inputs: &HashMap<String, Tensor>,
        options: &Options,
        config: &SessionConfig,
    ) -> Result<Self, SessionError> {
        let _ = env_logger::try_init();
        let model = Model::load_from_path(p).map_err(SessionError::ModelLoadError)?;
        let graph = model.graph;
        let input_ty: Vec<ResolvedTensorType> = graph
            .input_values()
            .iter()
            .map(|&id| -> Result<ResolvedTensorType, SessionError> {
                let name = &graph.values[id].name;
                let tensor = inputs
                    .get(name)
                    .ok_or_else(|| SessionError::OtherError(format!("missing input {name:?}")))?;
                Ok(ResolvedTensorType::new(
                    tensor.data.elem_type(),
                    tensor.dims.clone(),
                ))
            })
            .collect::<Result<_, _>>()?;
        Self::from_graph_inner(graph, Some(&input_ty), options, config)
    }

    fn from_graph_inner(
        mut graph: Graph,
        input_ty: Option<&[ResolvedTensorType]>,
        options: &Options,
        config: &SessionConfig,
    ) -> Result<Self, SessionError> {
        if let Some(input_ty) = input_ty {
            graph
                .resolve_input_types(input_ty)
                .map_err(SessionError::TypeError)?;
        }

        transform_graph(&mut graph, options, config);
        info!("Transformed");

        let tmp_dir = TempDir::with_prefix("my_model_")
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        info!("Build directory: {:?}", tmp_dir.path());
        let build_dir = PathBuf::from(tmp_dir.path());

        if options.save_transformed_model {
            let path = build_dir.join("transformed.onnx");
            crate::onnx::save::save_graph(&graph, &path);
            info!("Transformed model saved to {:?}", path);
        }

        let input_values = graph.input_values();
        let output_values = graph.output_values();
        let input_names: Vec<String> = input_values
            .iter()
            .map(|&id| graph.values[id].name.clone())
            .collect();
        let output_names: Vec<String> = output_values
            .iter()
            .map(|&id| graph.values[id].name.clone())
            .collect();
        let inputs_ty = get_argument_types(&graph, &input_values)?;
        let outputs_ty = get_argument_types(&graph, &output_values)?;
        let initializer_ids = graph.initializer_ids();
        let initializer_names: Vec<String> = initializer_ids
            .iter()
            .map(|&id| graph.values[id].name.clone())
            .collect();
        let mut file_cache: HashMap<PathBuf, Arc<File>> = HashMap::new();
        let initializer: Vec<InitializerSource> = initializer_ids
            .iter()
            .map(|&id| -> Result<InitializerSource, SessionError> {
                if let Some(t) = graph.get_inline_initializer(id) {
                    return Ok(InitializerSource::Inline(StrictTensor::from(t)));
                }
                let ext = graph.get_external_ref(id).expect("initializer missing");
                let file = match file_cache.get(&ext.path) {
                    Some(f) => Arc::clone(f),
                    None => {
                        let f = Arc::new(
                            File::open(&ext.path)
                                .map_err(ModelLoadError::FileRead)
                                .map_err(SessionError::ModelLoadError)?,
                        );
                        file_cache.insert(ext.path.clone(), Arc::clone(&f));
                        f
                    }
                };
                let length = ext.length.ok_or_else(|| {
                    SessionError::OtherError(
                        "external initializer without length is unsupported".to_string(),
                    )
                })?;
                Ok(InitializerSource::External {
                    file,
                    offset: ext.offset,
                    length,
                    elem_type: ext.elem_type,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut schedule = Schedule::new(graph, options.clone());
        let schedule_passes = create_schedule_passes(options);
        schedule_passes.run(&mut schedule);
        info!("Scheduled");

        let session_state_buffers: Vec<Arc<DeviceBuffer>> = {
            schedule
                .session_states
                .iter()
                .map(|&value_id| -> Result<Arc<DeviceBuffer>, SessionError> {
                    let name = &schedule.graph().values[value_id].name;
                    config
                        .session_states
                        .iter()
                        .find(|s| &s.name == name)
                        .map(|s| Arc::clone(&s.buffer))
                        .ok_or_else(|| {
                            SessionError::OtherError(format!(
                                "no DeviceBuffer provided for session state {name:?}"
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        if options.save_build_dir {
            let path = tmp_dir.keep();
            info!("Build directory saved at {:?}", path);
        }

        let use_hybrid_runtime = options.placement_strategy.is_some();
        let inner = if use_hybrid_runtime {
            SessionHybrid::new(
                inputs_ty,
                outputs_ty,
                initializer,
                session_state_buffers,
                schedule,
            )
            .map(SessionInner::Hybrid)?
        } else {
            match options.target {
                Target::CPU => {
                    if config.initializer_buffers.is_some() {
                        return Err(SessionError::OtherError(
                            "InitializerBuffers is not supported on CPU target".to_string(),
                        ));
                    }
                    SessionCPU::new(
                        inputs_ty,
                        outputs_ty,
                        initializer,
                        session_state_buffers,
                        schedule,
                        options,
                        &build_dir,
                    )
                    .map(SessionInner::CPU)?
                }
                Target::CUDA => SessionCUDA::new(
                    inputs_ty,
                    outputs_ty,
                    initializer,
                    initializer_names,
                    config.initializer_buffers.clone(),
                    session_state_buffers,
                    schedule,
                    options,
                    &build_dir,
                )
                .map(SessionInner::CUDA)?,
            }
        };
        Ok(Session {
            inner,
            input_names,
            output_names,
        })
    }

    // TODO: Type check
    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        match &mut self.inner {
            SessionInner::CPU(session) => session.run(inputs),
            SessionInner::CUDA(session) => session.run(inputs),
            SessionInner::Hybrid(session) => session.run(inputs),
        }
    }

    pub fn run_named(
        &mut self,
        mut inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, SessionError> {
        let inputs_vec: Vec<Tensor> = self
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .remove(name)
                    .ok_or_else(|| SessionError::OtherError(format!("missing input {name:?}")))
            })
            .collect::<Result<_, _>>()?;
        let outputs = self.run(&inputs_vec)?;
        Ok(self.output_names.iter().cloned().zip(outputs).collect())
    }
}
