//
// Created by Lazurite on 4/18/2022.
//

#ifndef PYBIND_CUDA_CPP_EXTENSION_H
#define PYBIND_CUDA_CPP_EXTENSION_H

#include <torch/extension.h>
#include <iostream>

extern void init_module_cpp_extension(pybind11::module &m);


#endif //PYBIND_CUDA_CPP_EXTENSION_H
