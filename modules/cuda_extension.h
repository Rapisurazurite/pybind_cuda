//
// Created by Lazurite on 4/19/2022.
//

#ifndef PYBIND_CUDA_CUDA_EXTENSION_H
#define PYBIND_CUDA_CUDA_EXTENSION_H

#include <torch/extension.h>
#include <iostream>

extern void init_module_cuda_extension(pybind11::module &m);

#endif //PYBIND_CUDA_CUDA_EXTENSION_H
