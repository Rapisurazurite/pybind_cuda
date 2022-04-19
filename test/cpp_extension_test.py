import import_helper
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pybind_cuda



class LLTMpy(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTMpy, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell

class LLTMFunctionCpp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = pybind_cuda.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = pybind_cuda.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTMcpp(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTMcpp, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunctionCpp.apply(input, self.weights, self.bias, *state)

def python_implementation():
    batch_size = 16
    input_features = 32
    state_size = 128

    # Note the device=cuda_device arguments here
    X = torch.randn(batch_size, input_features)
    h = torch.randn(batch_size, state_size)
    C = torch.randn(batch_size, state_size)

    rnn = LLTMpy(input_features, state_size)

    forward = 0
    backward = 0
    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start
    print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

def cpp_implementation():
    batch_size = 16
    input_features = 32
    state_size = 128

    # Note the device=cuda_device arguments here
    X = torch.randn(batch_size, input_features)
    h = torch.randn(batch_size, state_size)
    C = torch.randn(batch_size, state_size)

    rnn = LLTMcpp(input_features, state_size)

    forward = 0
    backward = 0
    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start
    print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

def python_implementation_cu():
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = 16
    input_features = 32
    state_size = 128

    # Note the device=cuda_device arguments here
    X = torch.randn(batch_size, input_features, device=cuda_device)
    h = torch.randn(batch_size, state_size, device=cuda_device)
    C = torch.randn(batch_size, state_size, device=cuda_device)

    rnn = LLTMpy(input_features, state_size).to(cuda_device)

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

def cpp_implementation_cu():
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = 16
    input_features = 32
    state_size = 128

    # Note the device=cuda_device arguments here
    X = torch.randn(batch_size, input_features, device=cuda_device)
    h = torch.randn(batch_size, state_size, device=cuda_device)
    C = torch.randn(batch_size, state_size, device=cuda_device)

    rnn = LLTMcpp(input_features, state_size).to(cuda_device)

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
    assert pybind_cuda.is_cpp_extension_init()

    python_implementation()
    cpp_implementation()

    python_implementation_cu()
    cpp_implementation_cu()

    # ??? This is so strange ???
    # Forward: 115.215 us | Backward 171.027 us
    # Forward: 99.602 us | Backward 282.254 us
    # Forward: 368.693 us | Backward 588.300 us
    # Forward: 311.143 us | Backward 942.109 us
