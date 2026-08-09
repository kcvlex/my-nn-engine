use super::ftype::FType;
use crate::collective;

/// Sum-all-reduce `num_elements` of `dtype` across the process group: copies
/// `src` into `dst` (unless they alias) and reduces in place through the
/// communicator registered via `collective::set_communicator`. Aborts if no
/// communicator is set or the collective fails; generated code has no error
/// path.
///
/// # Safety
/// `src` and `dst` are valid for `num_elements` of `dtype` and either alias
/// exactly or do not overlap.
#[no_mangle]
pub unsafe extern "C" fn mynn_all_reduce(
    dst: *mut u8,
    src: *const u8,
    num_elements: usize,
    dtype: FType,
) {
    let comm = collective::communicator()
        .expect("AllReduce kernel ran without a communicator; call collective::set_communicator");
    let comm_dtype = match dtype {
        FType::F32 => my_nn_engine_comm::DataType::F32,
        FType::Bf16 => panic!("AllReduce does not support bf16 yet"),
    };
    let num_bytes = num_elements * comm_dtype.size_of();
    if !std::ptr::eq(dst.cast_const(), src) {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, num_bytes) };
    }
    let buf = unsafe { std::slice::from_raw_parts_mut(dst, num_bytes) };
    comm.all_reduce(buf, comm_dtype, my_nn_engine_comm::ReduceOp::Sum)
        .expect("all_reduce failed");
}
