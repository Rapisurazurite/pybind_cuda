cmake_minimum_required(VERSION 3.16)
project(pybind_cuda)

find_package(CUDA REQUIRED)

include_directories("/home/ubuntu/anaconda3/envs/Openpcdet/lib/python3.8/site-packages/torch/include")
include_directories("/home/ubuntu/anaconda3/envs/Openpcdet/lib/python3.8/site-packages/torch/include/torch/csrc/api/include/")
set(PYTHON_EXECUTABLE /home/ubuntu/anaconda3/envs/Openpcdet/bin/python)
add_subdirectory(extern/pybind11)


## PROJECT pybind_cuda
set(module_name pybind_cuda)
set(pybind_cuda_SRCS main.cpp modules/cpp_extension.cpp)

pybind11_add_module(${module_name} ${pybind_cuda_SRCS})

# pytorch build
add_custom_target(
    pytorch_build
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake_setup.py build --module_name ${module_name} --src ${pybind_cuda_SRCS}
)
