use crate::codegen::cuda::*;
use crate::tensor::types::DataType;
use delegate::delegate;
use derive_more::From;

#[derive(From)]
pub enum CUDAKernel {
    MaxPoolKernel(MaxPoolKernel),
}

pub struct MaxPoolKernel {
    pub ty: DataType,

    pub out: Expr,
    pub in_: Expr,
    pub nbatch: Expr,
    pub channels: Expr,
    pub height: Expr,
    pub width: Expr,
    pub o_height: Expr,
    pub o_width: Expr,
    pub kernel_h: Expr,
    pub kernel_w: Expr,
    pub stride_h: Expr,
    pub stride_w: Expr,
    pub pad_h: Expr,
    pub pad_w: Expr,
}

impl MaxPoolKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        macro_rules! cast {
            ($e:expr) => {
                format!("({} *)({})", self.ty.fragment(), $e.fragment())
            };
        }
        let id = format!("max_pool_kernel<{}>", self.ty.fragment());
        let args = vec![
            cast!(self.out),
            cast!(self.in_),
            format!("std::numeric_limits<{}>::min()", self.ty.fragment()),
            self.nbatch.fragment(),
            self.channels.fragment(),
            self.height.fragment(),
            self.width.fragment(),
            self.o_height.fragment(),
            self.o_width.fragment(),
            self.kernel_h.fragment(),
            self.kernel_w.fragment(),
            self.stride_h.fragment(),
            self.stride_w.fragment(),
            self.pad_h.fragment(),
            self.pad_w.fragment(),
        ];
        (id, args)
    }
}

pub struct LaunchKernel {
    pub cuda_kernel: CUDAKernel,
    pub grid_size: Expr,
    pub block_size: Expr,
    pub shared_mem_bytes: Option<usize>,
    pub stream_id: StreamId,
}

impl LaunchKernel {
    delegate! {
        to match &self.cuda_kernel {
            CUDAKernel::MaxPoolKernel(m) => m,
        } {
            #[call(fragment)]
            fn kernel_fragment(&self) -> (String, Vec<String>);
        }
    }

    pub fn fragment(&self) -> String {
        let (id, args) = self.kernel_fragment();
        format!(
            "{id}<<<{grid_size}, {block_size}, {shared_mem_size}, {stream}>>>({args})",
            grid_size = self.grid_size.fragment(),
            block_size = self.block_size.fragment(),
            shared_mem_size = self.shared_mem_bytes.unwrap_or(0),
            stream = self.stream_id.to_identifier().fragment(),
            args = args.join(", ")
        )
    }
}
