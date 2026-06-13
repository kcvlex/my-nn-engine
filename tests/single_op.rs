mod common;

use itertools::izip;
use my_nn_engine::onnx::load::*;
use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine::schedule::ir::Device;
use my_nn_engine::schedule::scheduler::PlacementStrategy;
use my_nn_engine::session::Session;
use my_nn_engine::session::SessionConfig;
use my_nn_engine::session::SessionError;
use my_nn_engine::tensor::data::CompPolicy;
use my_nn_engine::tensor::Tensor;

#[derive(Clone, Copy)]
enum SessionKind {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    HybridCpu,
    #[cfg(feature = "cuda")]
    HybridCuda,
}

impl SessionKind {
    fn build_options(self) -> Options {
        match self {
            SessionKind::Cpu => Options::builder()
                .target(Target::CPU)
                .omp_elementwise_threshold(10)
                .build(),
            #[cfg(feature = "cuda")]
            SessionKind::Cuda => Options::builder().target(Target::CUDA).build(),
            SessionKind::HybridCpu => Options::builder()
                .target(Target::CPU)
                .omp_elementwise_threshold(10)
                .placement_strategy(Some(PlacementStrategy::Uniform(Device::CPU)))
                .build(),
            #[cfg(feature = "cuda")]
            SessionKind::HybridCuda => Options::builder()
                .target(Target::CUDA)
                .placement_strategy(Some(PlacementStrategy::Uniform(Device::CUDA)))
                .build(),
        }
    }
}

pub type TestResult = Result<(), SessionError>;

macro_rules! make_tensor {
    ($ty: ty, $($expr: expr,)*) => {{
        let orig: ndarray::Array<$ty, _> = ndarray::array!($($expr,)*);
        let res: Result<(Tensor, _), _> = orig
            .clone()
            .into_dyn()
            .try_into()
            .map(|t| (t, orig.clone()))
            .map_err(SessionError::TypeError);
        res
    }};
}

macro_rules! assert_eq_epsilon {
    ($left: expr, $right: expr, $epsilon: expr) => {{
        let res = $left.eq_with_epsilon(&$right, $epsilon, CompPolicy::Either);
        if !res {
            // For pretty print
            assert_eq!($left, $right);
        }
    }};
}

fn with_session<P, F>(p: P, kinds: &[SessionKind], f: F) -> TestResult
where
    P: AsRef<std::path::Path>,
    F: Fn(&mut Session) -> TestResult,
{
    use std::path::PathBuf;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/test/single_op")
        .join(p);
    for kind in kinds.iter().copied() {
        let opt = kind.build_options();
        let mut session = Session::new(&path, None, &opt, &SessionConfig::default())?;
        f(&mut session)?;
    }
    Ok(())
}

fn with_session_and_tensors<P, F>(
    dir: P,
    kinds: &[SessionKind],
    nums: (usize, usize),
    f: F,
) -> TestResult
where
    P: AsRef<std::path::Path>,
    F: Fn(&mut Session, (&[Tensor], &[Tensor])) -> TestResult,
{
    use std::path::PathBuf;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/test/single_op")
        .join(dir);
    let (input_num, output_num) = nums;
    let inputs = (0..input_num)
        .map(|i| {
            Tensor::load_from_path(dir.join(format!("input_{}.pb", i))).map_err(|e| {
                SessionError::OtherError(format!("Failed to load input {}: {:?}", i, e))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = (0..output_num)
        .map(|i| {
            Tensor::load_from_path(dir.join(format!("output_{}.pb", i))).map_err(|e| {
                SessionError::OtherError(format!("Failed to load output {}: {:?}", i, e))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    for kind in kinds.iter().copied() {
        let opt = kind.build_options();
        let mut session = Session::new(
            dir.join("model.onnx"),
            None,
            &opt,
            &SessionConfig::default(),
        )?;
        f(&mut session, (&inputs, &outputs))?;
    }
    Ok(())
}

fn with_cpu_session<P, F>(p: P, f: F) -> TestResult
where
    P: AsRef<std::path::Path>,
    F: Fn(&mut Session) -> TestResult,
{
    with_session(p, &[SessionKind::Cpu, SessionKind::HybridCpu], f)
}

fn with_all_sessions<P, F>(p: P, f: F) -> TestResult
where
    P: AsRef<std::path::Path>,
    F: Fn(&mut Session) -> TestResult,
{
    #[cfg(feature = "cuda")]
    let kinds = &[
        SessionKind::Cpu,
        SessionKind::Cuda,
        SessionKind::HybridCpu,
        SessionKind::HybridCuda,
    ];
    #[cfg(not(feature = "cuda"))]
    let kinds = &[SessionKind::Cpu, SessionKind::HybridCpu];
    with_session(p, kinds, f)
}

pub fn with_all_sessions_and_tensors<P, F>(p: P, nums: (usize, usize), f: F) -> TestResult
where
    P: AsRef<std::path::Path>,
    F: Fn(&mut Session, (&[Tensor], &[Tensor])) -> TestResult,
{
    #[cfg(feature = "cuda")]
    let kinds = &[
        SessionKind::Cpu,
        SessionKind::Cuda,
        SessionKind::HybridCpu,
        SessionKind::HybridCuda,
    ];
    #[cfg(not(feature = "cuda"))]
    let kinds = &[SessionKind::Cpu, SessionKind::HybridCpu];
    with_session_and_tensors(p, kinds, nums, f)
}

#[cfg(feature = "cuda")]
pub fn with_cuda_session_and_tensors<P, F>(p: P, nums: (usize, usize), f: F) -> TestResult
where
    P: AsRef<std::path::Path>,
    F: Fn(&mut Session, (&[Tensor], &[Tensor])) -> TestResult,
{
    with_session_and_tensors(p, &[SessionKind::Cuda], nums, f)
}

#[test]
fn add() -> TestResult {
    with_all_sessions_and_tensors("add", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn add_large() -> TestResult {
    with_all_sessions_and_tensors("add_large", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn add_broadcast() -> TestResult {
    with_all_sessions_and_tensors("add_broadcast", (3, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn relu() -> TestResult {
    with_all_sessions_and_tensors("relu", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn clip() -> TestResult {
    with_all_sessions_and_tensors("clip", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn transpose() -> TestResult {
    with_all_sessions_and_tensors("transpose", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn matmul() -> TestResult {
    with_all_sessions_and_tensors("matmul", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn matmul_a_x_tb() -> TestResult {
    with_all_sessions_and_tensors("matmul_a_x_tb", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn conv() -> TestResult {
    with_all_sessions_and_tensors("conv", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn conv_with_strides0() -> TestResult {
    with_all_sessions_and_tensors(
        "conv_with_strides0",
        (2, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

#[test]
fn conv_with_strides1() -> TestResult {
    with_all_sessions_and_tensors(
        "conv_with_strides1",
        (2, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

#[test]
fn conv_with_strides2() -> TestResult {
    with_all_sessions_and_tensors(
        "conv_with_strides2",
        (2, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

#[test]
fn conv_channels() -> TestResult {
    with_all_sessions_and_tensors("conv_channels", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn conv_with_autopad_same() -> TestResult {
    with_all_sessions_and_tensors(
        "conv_with_autopad_same",
        (2, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

#[test]
fn conv_bias() -> TestResult {
    with_all_sessions_and_tensors("conv_bias", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn depthwise_conv() -> TestResult {
    with_all_sessions_and_tensors("depthwise_conv", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn depthwise_conv_bias() -> TestResult {
    with_all_sessions_and_tensors(
        "depthwise_conv_bias",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

#[test]
fn avgpool() -> TestResult {
    with_all_sessions_and_tensors("avgpool", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn avgpool_no_pad() -> TestResult {
    with_all_sessions_and_tensors("avgpool_no_pad", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn maxpool() -> TestResult {
    with_all_sessions_and_tensors("maxpool", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn reducemax() -> TestResult {
    with_all_sessions_and_tensors("reducemax", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn large_global_avg() -> TestResult {
    with_all_sessions_and_tensors("large_global_avg", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-2);
        Ok(())
    })
}

#[test]
fn global_avg_non_pow2() -> TestResult {
    with_all_sessions_and_tensors(
        "global_avg_non_pow2",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-2);
            Ok(())
        },
    )
}

#[test]
fn batchnorm() -> TestResult {
    with_all_sessions_and_tensors("batchnorm", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-3);
        Ok(())
    })
}

#[test]
fn leaky_relu() -> TestResult {
    with_all_sessions_and_tensors("leakyrelu", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn exp() -> TestResult {
    with_all_sessions_and_tensors("exp", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn log() -> TestResult {
    with_all_sessions_and_tensors("log", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn tanh() -> TestResult {
    with_all_sessions_and_tensors("tanh", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn sigmoid() -> TestResult {
    with_all_sessions_and_tensors("sigmoid", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn swish() -> TestResult {
    with_all_sessions_and_tensors("swish", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn resize_downsample_sizes_nearest() -> TestResult {
    with_all_sessions_and_tensors(
        "resize_downsample_sizes_nearest",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn resize_upsample_scales_nearest() -> TestResult {
    with_all_sessions_and_tensors(
        "resize_upsample_scales_nearest",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn resize_upsample_scales_nearest_axes_2_3() -> TestResult {
    with_all_sessions_and_tensors(
        "resize_upsample_scales_nearest_axes_2_3",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn resize_upsample_scales_nearest_axes_3_2() -> TestResult {
    with_all_sessions_and_tensors(
        "resize_upsample_scales_nearest_axes_3_2",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn resize_upsample_sizes_nearest_axes_2_3() -> TestResult {
    with_all_sessions_and_tensors(
        "resize_upsample_sizes_nearest_axes_2_3",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn resize_upsample_sizes_nearest_axes_3_2() -> TestResult {
    with_all_sessions_and_tensors(
        "resize_upsample_sizes_nearest_axes_3_2",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn resize_upsample_sizes_nearest_ceil_half_pixel() -> TestResult {
    with_all_sessions_and_tensors(
        "resize_upsample_sizes_nearest_ceil_half_pixel",
        (1, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn slice_large_end() -> TestResult {
    with_session_and_tensors(
        "slice_large_end",
        &[SessionKind::Cpu, SessionKind::HybridCpu],
        (0, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
            Ok(())
        },
    )
}

#[test]
fn split_axis_2() -> TestResult {
    with_all_sessions("split_axis_2.onnx", |session| {
        let (input, _) = make_tensor!(
            f32,
            [[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]],
        )?;
        let (expected0, _) = make_tensor!(f32, [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],]],)?;
        let (expected1, _) = make_tensor!(f32, [[[9.0, 10.0, 11.0, 12.0],]],)?;
        let (expected2, _) = make_tensor!(f32, [[[13.0, 14.0, 15.0, 16.0],]],)?;
        let output = session.run(&[input])?;
        assert_eq!(output[0], expected0);
        assert_eq!(output[1], expected1);
        assert_eq!(output[2], expected2);
        Ok(())
    })
}

#[test]
fn split_axis_3() -> TestResult {
    with_all_sessions("split_axis_3.onnx", |session| {
        let (input, _) = make_tensor!(
            f32,
            [[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]],
        )?;
        let (expected0, _) =
            make_tensor!(f32, [[[1.0, 2.0], [5.0, 6.0], [9.0, 10.0], [13.0, 14.0],]],)?;
        let (expected1, _) = make_tensor!(f32, [[[3.0], [7.0], [11.0], [15.0],]],)?;
        let (expected2, _) = make_tensor!(f32, [[[4.0], [8.0], [12.0], [16.0],]],)?;
        let output = session.run(&[input])?;
        assert_eq!(output[0], expected0);
        assert_eq!(output[1], expected1);
        assert_eq!(output[2], expected2);
        Ok(())
    })
}

#[test]
fn concat_axis_2() -> TestResult {
    with_all_sessions_and_tensors("concat_axis_2", (3, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn cast_f32_to_i64() -> TestResult {
    with_all_sessions_and_tensors("cast_f32_to_i64", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq!(outputs[0], expected[0]);
        Ok(())
    })
}

#[test]
fn bias_gemm() -> TestResult {
    with_all_sessions_and_tensors("bias_gemm", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn squeeze() -> TestResult {
    with_all_sessions_and_tensors("squeeze", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn squeeze_opt() -> TestResult {
    with_all_sessions_and_tensors("squeeze_opt", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn squeeze_scalar() -> TestResult {
    // Minimal reproducer for Squeeze bug found in bertsquad-12 node 1148
    // Input shape: (1, 1, 1), Squeeze all dims -> Output: () (scalar)
    // Bug: Implementation crashes with SIGSEGV when producing scalar output
    // This tests that Squeeze handles zero-dimensional tensors correctly
    with_all_sessions_and_tensors("squeeze_scalar", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn unsqueeze() -> TestResult {
    with_all_sessions_and_tensors("unsqueeze", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn reciprocal() -> TestResult {
    with_all_sessions_and_tensors("reciprocal", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn sqrt() -> TestResult {
    with_all_sessions_and_tensors("sqrt", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn pow_types() -> TestResult {
    let (x_i32, _) = make_tensor!(i32, 1, 2, 3,)?;
    let (x_i64, _) = make_tensor!(i64, 1, 2, 3,)?;
    let (x_u64, _) = make_tensor!(u64, 1, 2, 3,)?;
    let (x_f32, _) = make_tensor!(f32, 1.0, 2.0, 3.0,)?;
    let (x_f64, _) = make_tensor!(f64, 1.0, 2.0, 3.0,)?;

    let (y_i32, _) = make_tensor!(i32, 4, 5, 6,)?;
    let (y_i64, _) = make_tensor!(i64, 4, 5, 6,)?;
    let (y_u64, _) = make_tensor!(u64, 4, 5, 6,)?;
    let (y_f32, _) = make_tensor!(f32, 4.0, 5.0, 6.0,)?;
    let (y_f64, _) = make_tensor!(f64, 4.0, 5.0, 6.0,)?;

    let (z_i32, _) = make_tensor!(i32, 1, 32, 729,)?;
    let (z_i64, _) = make_tensor!(i64, 1, 32, 729,)?;
    let (z_u64, _) = make_tensor!(u64, 1, 32, 729,)?;
    let (z_f32, _) = make_tensor!(f32, 1.0, 32.0, 729.0,)?;
    let (z_f64, _) = make_tensor!(f64, 1.0, 32.0, 729.0,)?;

    let xv = [x_i32, x_i64, x_u64, x_f32, x_f64];
    let yv = [y_i32, y_i64, y_u64, y_f32, y_f64];
    let zv = [z_i32, z_i64, z_u64, z_f32, z_f64];
    let ty_lit = ["i32", "i64", "u64", "f32", "f64"];

    for (x, z, lty) in izip!(xv, zv, ty_lit) {
        for (y, rty) in izip!(yv.iter(), ty_lit) {
            let onnx = format!("pow_{}_{}.onnx", lty, rty);

            // TODO: Support integer types for CUDA.
            if lty.starts_with('f') && rty.starts_with('f') {
                with_all_sessions(onnx, |session| {
                    let output = session.run(&[x.clone(), y.clone()])?;
                    assert_eq_epsilon!(output[0], z.clone(), 1e-6);
                    Ok(())
                })?;
            } else {
                with_cpu_session(onnx, |session| {
                    let output = session.run(&[x.clone(), y.clone()])?;
                    assert_eq!(output[0], z.clone());
                    Ok(())
                })?;
            }
        }
    }

    Ok(())
}

#[test]
fn sub() -> TestResult {
    with_all_sessions_and_tensors("sub", (3, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn constantofshape_float_ones() -> TestResult {
    with_all_sessions_and_tensors(
        "constantofshape_float_ones",
        (0, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn softmax() -> TestResult {
    with_all_sessions_and_tensors("softmax", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-2);
        Ok(())
    })
}

#[test]
fn softmax2() -> TestResult {
    with_all_sessions_and_tensors("softmax2", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn softmax_axis() -> TestResult {
    with_all_sessions_and_tensors("softmax_axis", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn one_hot() -> TestResult {
    with_all_sessions_and_tensors("one_hot", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn batched_gemm() -> TestResult {
    with_all_sessions_and_tensors("batched_gemm", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1.0);
        Ok(())
    })
}

#[test]
fn gather_default_axis() -> TestResult {
    with_all_sessions_and_tensors(
        "gather_default_axis",
        (2, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 0.0);
            Ok(())
        },
    )
}

#[test]
fn gather_dim1() -> TestResult {
    // Reproducer for zero-stride bug: indices shape [1, N] causes stride[0]=0
    // Expected: [[3,4,5], [6,7,8], [0,1,2], [9,10,11]]
    // Bug: All zeros or garbage due to always reading from same offset
    // CPU-only test (CUDA has separate i32 type handling issues)
    with_session_and_tensors(
        "gather_dim1",
        &[SessionKind::Cpu, SessionKind::HybridCpu],
        (2, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 0.0);
            Ok(())
        },
    )
}

#[test]
fn gather_negative_indices() -> TestResult {
    // Reproducer for BERT crash: negative indices like -8, -2, -25
    // In ONNX/NumPy, negative indices mean "from end": -1 is last, -2 is second-to-last
    with_session_and_tensors(
        "gather_negative_indices",
        &[SessionKind::Cpu, SessionKind::HybridCpu],
        (2, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
            Ok(())
        },
    )
}

#[test]
fn non_zero() -> TestResult {
    with_all_sessions_and_tensors("non_zero", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 0.0);
        Ok(())
    })
}

#[test]
fn layer_norm() -> TestResult {
    with_all_sessions_and_tensors("layer_norm", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1.0);
        Ok(())
    })
}

#[test]
fn gelu_tanh() -> TestResult {
    with_all_sessions_and_tensors("gelu_tanh", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn attention_no_causal() -> TestResult {
    with_all_sessions_and_tensors(
        "attention_no_causal",
        (3, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

#[test]
fn attention_causal() -> TestResult {
    with_all_sessions_and_tensors("attention_causal", (3, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn attention_causal_large() -> TestResult {
    with_all_sessions_and_tensors(
        "attention_causal_large",
        (3, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

#[test]
fn attention_decode() -> TestResult {
    with_all_sessions_and_tensors("attention_decode", (3, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn kv_cache_update() -> TestResult {
    with_all_sessions_and_tensors("kv_cache_update", (3, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn attention_decode_gqa() -> TestResult {
    with_all_sessions_and_tensors(
        "attention_decode_gqa",
        (3, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

#[test]
fn attention_decode_runtime_kv() -> TestResult {
    with_all_sessions_and_tensors(
        "attention_decode_runtime_kv",
        (4, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
            Ok(())
        },
    )
}

// Transpose generates Contiguous with non-empty ops after FoldContiguous:
// Transpose -> Output => Reinterpret -> Contiguous -> Output => Contiguous(transpose) -> Output
#[test]
fn transpose_contiguous_fold() -> TestResult {
    with_all_sessions_and_tensors("transpose", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-5);
        Ok(())
    })
}

#[test]
fn cos() -> TestResult {
    with_all_sessions_and_tensors("cos", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn sin() -> TestResult {
    with_all_sessions_and_tensors("sin", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn conv_stride2_1x1() -> TestResult {
    with_all_sessions_and_tensors("conv_stride2_1x1", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-4);
        Ok(())
    })
}

#[test]
fn conv_3x3_stride2() -> TestResult {
    with_all_sessions_and_tensors("conv_3x3_stride2", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-3);
        Ok(())
    })
}

#[test]
fn neg() -> TestResult {
    with_all_sessions_and_tensors("neg", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn range() -> TestResult {
    with_session_and_tensors(
        "range",
        &[SessionKind::Cpu, SessionKind::HybridCpu],
        (0, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq!(outputs[0], expected[0]);
            Ok(())
        },
    )
}

#[test]
fn r#where() -> TestResult {
    with_all_sessions_and_tensors("where", (3, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-6);
        Ok(())
    })
}

#[test]
fn equal() -> TestResult {
    with_all_sessions_and_tensors("equal", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq!(outputs[0], expected[0]);
        Ok(())
    })
}

#[test]
fn expand() -> TestResult {
    with_all_sessions_and_tensors("expand", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq!(outputs[0], expected[0]);
        Ok(())
    })
}

#[test]
fn flatten() -> TestResult {
    with_all_sessions_and_tensors("flatten", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq!(outputs[0], expected[0]);
        Ok(())
    })
}

#[test]
fn isnan() -> TestResult {
    with_all_sessions_and_tensors("isnan", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq!(outputs[0], expected[0]);
        Ok(())
    })
}

#[test]
fn and() -> TestResult {
    with_all_sessions_and_tensors("and", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq!(outputs[0], expected[0]);
        Ok(())
    })
}

#[test]
fn lessorequal() -> TestResult {
    with_all_sessions_and_tensors("lessorequal", (2, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq!(outputs[0], expected[0]);
        Ok(())
    })
}

#[test]
fn slice() -> TestResult {
    with_all_sessions_and_tensors("slice", (1, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq!(outputs[0], expected[0]);
        Ok(())
    })
}

#[test]
fn dequantize_linear() -> TestResult {
    with_all_sessions_and_tensors(
        "dequantize_linear",
        (2, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq_epsilon!(outputs[0], expected[0], 1e-2);
            Ok(())
        },
    )
}

macro_rules! dyn_quantize_linear_test {
    ($name:ident, $fixture:literal, $run:path) => {
        #[test]
        fn $name() -> TestResult {
            $run($fixture, (1, 3), |session, (inputs, expected)| {
                let outputs = session.run(inputs)?;
                assert_eq!(outputs.len(), 3);
                assert_eq!(outputs[0], expected[0], "y mismatch");
                assert_eq_epsilon!(outputs[1], expected[1], 1e-3);
                assert_eq!(outputs[2], expected[2], "zero_point mismatch");
                Ok(())
            })
        }
    };
}

// Symmetric per-row also runs on CPU (the mynn_dynquant_i8 kernel); per-tensor
// and asymmetric variants are CUDA-only for now.
dyn_quantize_linear_test!(
    dyn_quantize_linear_sym_per_row,
    "dyn_quantize_linear_sym_per_row",
    with_all_sessions_and_tensors
);
#[cfg(feature = "cuda")]
dyn_quantize_linear_test!(
    dyn_quantize_linear_sym_per_tensor,
    "dyn_quantize_linear_sym_per_tensor",
    with_cuda_session_and_tensors
);
#[cfg(feature = "cuda")]
dyn_quantize_linear_test!(
    dyn_quantize_linear_asym_per_tensor,
    "dyn_quantize_linear_asym_per_tensor",
    with_cuda_session_and_tensors
);
#[cfg(feature = "cuda")]
dyn_quantize_linear_test!(
    dyn_quantize_linear_asym_per_row,
    "dyn_quantize_linear_asym_per_row",
    with_cuda_session_and_tensors
);

macro_rules! quantized_matmul_test {
    ($name:ident, $fixture:literal) => {
        #[test]
        fn $name() -> TestResult {
            with_all_sessions_and_tensors($fixture, (4, 1), |session, (inputs, expected)| {
                let outputs = session.run(inputs)?;
                assert_eq_epsilon!(outputs[0], expected[0], 1e-1);
                Ok(())
            })
        }
    };
}

quantized_matmul_test!(quantized_matmul_64x64x64, "quantized_matmul_64x64x64");
quantized_matmul_test!(quantized_matmul_128x128x128, "quantized_matmul_128x128x128");
quantized_matmul_test!(quantized_matmul_256x128x128, "quantized_matmul_256x128x128");
quantized_matmul_test!(quantized_matmul_128x256x64, "quantized_matmul_128x256x64");
quantized_matmul_test!(quantized_matmul_1x128x128, "quantized_matmul_1x128x128");

#[test]
fn dequant_matmul() -> TestResult {
    with_all_sessions_and_tensors("dequant_matmul", (3, 1), |session, (inputs, expected)| {
        let outputs = session.run(inputs)?;
        assert_eq_epsilon!(outputs[0], expected[0], 1e-1);
        Ok(())
    })
}

macro_rules! dequant_matmul_size_test {
    ($name:ident, $fixture:literal) => {
        #[test]
        fn $name() -> TestResult {
            with_all_sessions_and_tensors($fixture, (3, 1), |session, (inputs, expected)| {
                let outputs = session.run(inputs)?;
                assert_eq_epsilon!(outputs[0], expected[0], 1e-1);
                Ok(())
            })
        }
    };
}

dequant_matmul_size_test!(dequant_matmul_16x16x16, "dequant_matmul_16x16x16");
dequant_matmul_size_test!(dequant_matmul_16x32x32, "dequant_matmul_16x32x32");
dequant_matmul_size_test!(dequant_matmul_32x32x64, "dequant_matmul_32x32x64");
dequant_matmul_size_test!(dequant_matmul_64x64x128, "dequant_matmul_64x64x128");
dequant_matmul_size_test!(dequant_matmul_128x128x64, "dequant_matmul_128x128x64");
dequant_matmul_size_test!(dequant_matmul_24x48x40, "dequant_matmul_24x48x40");
dequant_matmul_size_test!(dequant_matmul_17x17x17, "dequant_matmul_17x17x17");
dequant_matmul_size_test!(dequant_matmul_50x50x50, "dequant_matmul_50x50x50");
dequant_matmul_size_test!(dequant_matmul_80x96x96, "dequant_matmul_80x96x96");

#[test]
fn quantizing_kvcache_update() -> TestResult {
    with_all_sessions_and_tensors(
        "quantizing_kvcache_update",
        (4, 1),
        |session, (inputs, expected)| {
            let outputs = session.run(inputs)?;
            assert_eq!(outputs[0], expected[0]);
            Ok(())
        },
    )
}
