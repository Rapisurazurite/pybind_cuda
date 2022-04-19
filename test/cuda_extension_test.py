import import_helper
import math
import time
import torch
import torch.nn as nn
from torch.autograd import Function
import pybind_cuda_cu as pybind_cuda


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = pybind_cuda.forward_cu(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = pybind_cuda.backward_cu(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)

def cu_implementation():
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = 16
    input_features = 32
    state_size = 128

    # Note the device=cuda_device arguments here
    X = torch.randn(batch_size, input_features, device=cuda_device)
    h = torch.randn(batch_size, state_size, device=cuda_device)
    C = torch.randn(batch_size, state_size, device=cuda_device)

    rnn = LLTM(input_features, state_size).to(cuda_device)

    forward = 0
    backward = 0
    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        torch.cuda.synchronize()
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        torch.cuda.synchronize()
        backward += time.time() - start
    print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

if __name__ == '__main__':
    cu_implementation()
    # Forward: 282.260 us | Backward 504.928 us