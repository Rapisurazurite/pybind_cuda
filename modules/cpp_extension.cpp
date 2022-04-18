//
// Created by Lazurite on 4/18/2022.
//

#include "cpp_extension.h"
#include <vector>


at::Tensor d_sigmoid(at::Tensor z) {
    auto s = at::sigmoid(z);
    return (1-s)*s;
}

std::vector<at::Tensor> lltm_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell){
    // X = torch.cat([old_h, input], dim=1)
    auto X = torch::cat({input, old_h}, 1);
    // gate_weights = F.linear(X, self.weights, self.bias)
    auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
    // gates = gate_weights.chunk(3, dim=1)
    auto gate = gate_weights.chunk(3, 1);
    // input_gate = torch.sigmoid(gates[0])
    auto input_gate = torch::sigmoid(gate[0]);
    // output_gate = torch.sigmoid(gates[1])
    auto output_gate = torch::sigmoid(gate[1]);
    // candidate_cell = F.elu(gates[2])
    auto candidate_cell = torch::elu(gate[2], 1.0);
    // new_cell = old_cell + candidate_cell * input_gate
    auto new_cell = old_cell + candidate_cell * input_gate;
    // new_h = torch.tanh(new_cell) * output_gate
    auto new_h = torch::tanh(new_cell) * output_gate;

    return {new_h,
            new_cell,
            input_gate,
            output_gate,
            candidate_cell,
            X,
            gate_weights};
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
    return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
    auto e = z.exp();
    auto mask = (alpha * (e - 1)) < 0;
    return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_backward(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gate_weights,
        torch::Tensor weights) {
    auto d_output_gate = torch::tanh(new_cell) * grad_h;
    auto d_tanh_new_cell = output_gate * grad_h;
    auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

    auto d_old_cell = d_new_cell;
    auto d_candidate_cell = input_gate * d_new_cell;
    auto d_input_gate = candidate_cell * d_new_cell;

    auto gates = gate_weights.chunk(3, /*dim=*/1);
    d_input_gate *= d_sigmoid(gates[0]);
    d_output_gate *= d_sigmoid(gates[1]);
    d_candidate_cell *= d_elu(gates[2]);

    auto d_gates =
            torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

    auto d_weights = d_gates.t().mm(X);
    auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

    auto d_X = d_gates.mm(weights);
    const auto state_size = grad_h.size(1);
    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}

void init_module_cpp_extension(pybind11::module &m){
    m.def("is_cpp_extension_init", [](){
        return true;
    });
    m.def("forward", &lltm_forward, "LLTM forward");
    m.def("backward", &lltm_backward, "LLTM backward");
}