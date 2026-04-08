#ifndef INCLUDE_CUDNN_SETTING_H_
#define INCLUDE_CUDNN_SETTING_H_

#include <cudnn.h>
#include <cstddef>
#include <cassert>
#include <vector>
#include <algorithm>

#define CUDNN_BE_CHECK(expr) do { auto status = (expr); assert(status == CUDNN_STATUS_SUCCESS); } while(0)

struct CudnnHandlerContext {
    cudnnHandle_t handle;
};

struct CudnnConvSetting {
    cudnnBackendDescriptor_t plan = nullptr;
    int64_t workspace_size = 0;
    void *workspace = nullptr;

    int64_t x_uid = 1, w_uid = 2, y_uid = 3, bias_uid = 4;

    ~CudnnConvSetting() {
        if (plan) cudnnBackendDestroyDescriptor(plan);
    }

    void build(
        CudnnHandlerContext *ctx,
        cudnnDataType_t data_type,
        int64_t n, int64_t c, int64_t h, int64_t w,
        int64_t k, int64_t r, int64_t s,
        int64_t out_h, int64_t out_w,
        int64_t pad_h_pre, int64_t pad_w_pre,
        int64_t pad_h_post, int64_t pad_w_post,
        int64_t stride_h, int64_t stride_w,
        int64_t dilation_h, int64_t dilation_w,
        int64_t groups,
        bool has_bias,
        bool has_relu,
        bool x_is_nhwc,
        bool y_is_nhwc
    ) {
        auto make_desc = [](cudnnBackendDescriptorType_t type) {
            cudnnBackendDescriptor_t d;
            CUDNN_BE_CHECK(cudnnBackendCreateDescriptor(type, &d));
            return d;
        };

        auto finalize = [](cudnnBackendDescriptor_t d) {
            CUDNN_BE_CHECK(cudnnBackendFinalize(d));
        };

        auto set_attr = [](cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name,
                          cudnnBackendAttributeType_t type, int64_t count, const void *data) {
            CUDNN_BE_CHECK(cudnnBackendSetAttribute(d, name, type, count, data));
        };

        int64_t alignment = 16;
        int64_t c_per_group = c / groups;

        std::vector<int64_t> x_dim = {n, c, h, w};
        std::vector<int64_t> x_str = x_is_nhwc
            ? std::vector<int64_t>{c*h*w, 1, w*c, c}
            : std::vector<int64_t>{c*h*w, h*w, w, 1};

        std::vector<int64_t> w_dim = {k, c_per_group, r, s};
        std::vector<int64_t> w_str = {c_per_group*r*s, r*s, s, 1};

        std::vector<int64_t> y_dim = {n, k, out_h, out_w};
        std::vector<int64_t> y_str = y_is_nhwc
            ? std::vector<int64_t>{k*out_h*out_w, 1, out_w*k, k}
            : std::vector<int64_t>{k*out_h*out_w, out_h*out_w, out_w, 1};

        auto make_tensor = [&](int64_t uid, std::vector<int64_t> &dim, std::vector<int64_t> &str, bool is_virtual = false) {
            auto d = make_desc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
            int64_t ndim = dim.size();
            set_attr(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &data_type);
            set_attr(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, ndim, dim.data());
            set_attr(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, ndim, str.data());
            set_attr(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
            set_attr(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
            set_attr(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
            finalize(d);
            return d;
        };

        auto x_desc = make_tensor(x_uid, x_dim, x_str);
        auto w_desc = make_tensor(w_uid, w_dim, w_str);

        bool conv_output_virtual = has_bias || has_relu;
        int64_t conv_out_uid = conv_output_virtual ? 100 : y_uid;
        auto conv_out_desc = make_tensor(conv_out_uid, y_dim, y_str, conv_output_virtual);

        std::vector<int64_t> pre_pad = {pad_h_pre, pad_w_pre};
        std::vector<int64_t> post_pad = {pad_h_post, pad_w_post};
        std::vector<int64_t> strides = {stride_h, stride_w};
        std::vector<int64_t> dilations = {dilation_h, dilation_w};
        int64_t ndim_spatial = 2;
        cudnnDataType_t compute_type = data_type;
        cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;

        auto conv_desc = make_desc(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        set_attr(conv_desc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute_type);
        set_attr(conv_desc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &conv_mode);
        set_attr(conv_desc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &ndim_spatial);
        set_attr(conv_desc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pre_pad.data());
        set_attr(conv_desc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, post_pad.data());
        set_attr(conv_desc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dilations.data());
        set_attr(conv_desc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, strides.data());
        finalize(conv_desc);

        float alpha = 1.0f, beta = 0.0f;
        auto fprop_desc = make_desc(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        set_attr(fprop_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_desc);
        set_attr(fprop_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_desc);
        set_attr(fprop_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_out_desc);
        set_attr(fprop_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_desc);
        set_attr(fprop_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
        set_attr(fprop_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
        finalize(fprop_desc);

        std::vector<cudnnBackendDescriptor_t> ops = {fprop_desc};
        auto current_desc = conv_out_desc;
        int64_t next_uid = 101;

        if (has_bias) {
            std::vector<int64_t> b_dim = {1, k, 1, 1};
            std::vector<int64_t> b_str = {k, 1, 1, 1};
            auto b_desc = make_tensor(bias_uid, b_dim, b_str);

            bool bias_out_virtual = has_relu;
            int64_t bias_out_uid = bias_out_virtual ? next_uid++ : y_uid;
            auto bias_out_desc = make_tensor(bias_out_uid, y_dim, y_str, bias_out_virtual);

            cudnnPointwiseMode_t add_mode = CUDNN_POINTWISE_ADD;
            auto pw_add_desc = make_desc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
            set_attr(pw_add_desc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &add_mode);
            set_attr(pw_add_desc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &compute_type);
            finalize(pw_add_desc);

            auto pw_add_op = make_desc(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
            set_attr(pw_add_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pw_add_desc);
            set_attr(pw_add_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &current_desc);
            set_attr(pw_add_op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b_desc);
            set_attr(pw_add_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bias_out_desc);
            set_attr(pw_add_op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &alpha);
            set_attr(pw_add_op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, CUDNN_TYPE_FLOAT, 1, &alpha);
            finalize(pw_add_op);

            ops.push_back(pw_add_op);
            current_desc = bias_out_desc;
        }

        if (has_relu) {
            auto relu_out_desc = make_tensor(y_uid, y_dim, y_str);

            cudnnPointwiseMode_t relu_mode = CUDNN_POINTWISE_RELU_FWD;
            auto pw_relu_desc = make_desc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
            set_attr(pw_relu_desc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &relu_mode);
            set_attr(pw_relu_desc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &compute_type);
            finalize(pw_relu_desc);

            auto pw_relu_op = make_desc(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
            set_attr(pw_relu_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pw_relu_desc);
            set_attr(pw_relu_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &current_desc);
            set_attr(pw_relu_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &relu_out_desc);
            set_attr(pw_relu_op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &alpha);
            finalize(pw_relu_op);

            ops.push_back(pw_relu_op);
        }

        auto op_graph = make_desc(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set_attr(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, ops.size(), ops.data());
        set_attr(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &ctx->handle);
        finalize(op_graph);

        auto engine_heur = make_desc(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        cudnnBackendHeurMode_t heur_mode = CUDNN_HEUR_MODE_A;
        set_attr(engine_heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph);
        set_attr(engine_heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heur_mode);
        finalize(engine_heur);

        int64_t engine_count = 0;
        CUDNN_BE_CHECK(cudnnBackendGetAttribute(engine_heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &engine_count, nullptr));
        assert(engine_count > 0);
        std::vector<cudnnBackendDescriptor_t> engine_configs(engine_count);
        for (auto &ec : engine_configs) CUDNN_BE_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &ec));
        CUDNN_BE_CHECK(cudnnBackendGetAttribute(engine_heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, engine_count, &engine_count, engine_configs.data()));

        plan = make_desc(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        set_attr(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &ctx->handle);
        set_attr(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_configs[0]);
        finalize(plan);

        CUDNN_BE_CHECK(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, nullptr, &workspace_size));

        for (auto &ec : engine_configs) cudnnBackendDestroyDescriptor(ec);
        cudnnBackendDestroyDescriptor(engine_heur);
        cudnnBackendDestroyDescriptor(op_graph);
        for (auto &op : ops) cudnnBackendDestroyDescriptor(op);
        cudnnBackendDestroyDescriptor(conv_desc);
        cudnnBackendDestroyDescriptor(x_desc);
        cudnnBackendDestroyDescriptor(w_desc);
        if (conv_output_virtual || current_desc != conv_out_desc)
            cudnnBackendDestroyDescriptor(conv_out_desc);
    }

    void execute(CudnnHandlerContext *ctx, void *x, void *w, void *y, void *bias) {
        std::vector<int64_t> uids = {x_uid, w_uid, y_uid};
        std::vector<void*> ptrs = {x, w, y};
        if (bias) {
            uids.push_back(bias_uid);
            ptrs.push_back(bias);
        }

        auto var_pack_desc = [&]() {
            cudnnBackendDescriptor_t d;
            CUDNN_BE_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &d));
            CUDNN_BE_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, ptrs.size(), ptrs.data()));
            CUDNN_BE_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data()));
            CUDNN_BE_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace));
            CUDNN_BE_CHECK(cudnnBackendFinalize(d));
            return d;
        }();

        CUDNN_BE_CHECK(cudnnBackendExecute(ctx->handle, plan, var_pack_desc));
        cudnnBackendDestroyDescriptor(var_pack_desc);
    }
};

#endif
