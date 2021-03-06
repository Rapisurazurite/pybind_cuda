//
// Created by Lazurite on 4/19/2022.
//

#include "cuda_extension.h"

#include <vector>
// CUDA forward declarations

std::vector<torch::Tensor> lltm_cuda_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_cuda_backward(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gate_weights,
        torch::Tensor weights);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> lltm_forward_cu(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(old_h);
    CHECK_INPUT(old_cell);

    return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> lltm_backward_cu(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gate_weights,
        torch::Tensor weights) {
    CHECK_INPUT(grad_h);
    CHECK_INPUT(grad_cell);
    CHECK_INPUT(input_gate);
    CHECK_INPUT(output_gate);
    CHECK_INPUT(candidate_cell);
    CHECK_INPUT(X);
    CHECK_INPUT(gate_weights);
    CHECK_INPUT(weights);

    return lltm_cuda_backward(
            grad_h,
            grad_cell,
            new_cell,
            input_gate,
            output_gate,
            candidate_cell,
            X,
            gate_weights,
            weights);
}

void init_module_cuda_extension(pybind11::module &m) {
    m.def("forward_cu", &lltm_forward_cu, "LLTM forward (CUDA)");
    m.def("backward_cu", &lltm_backward_cu, "LLTM backward (CUDA)");
}