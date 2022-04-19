cmake_minimum_required(VERSION 3.16)
project(pybind_cuda)

find_package(CUDA REQUIRED)


set(PYTHON_EXECUTABLE /home/ubuntu/anaconda3/envs/Openpcdet/bin/python3.8)
include_directories("/home/ubuntu/anaconda3/envs/Openpcdet/lib/python3.8/site-packages/torch/include")
include_directories("/home/ubuntu/anaconda3/envs/Openpcdet/lib/python3.8/site-packages/torch/include/torch/csrc/api/include/")

add_subdirectory(extern/pybind11)


## PROJECT pybind_cuda
set(module_name pybind_cuda)
set(pybind_cuda_SRCS main.cpp modules/cpp_extension.cpp modules/cuda_extension.cpp modules/cuda_kernel.cu)

pybind11_add_module(${module_name} ${pybind_cuda_SRCS})
add_custom_target(
    pytorch_build_cu
    COMMAND chmod a+x ${CMAKE_SOURCE_DIR}/cmake_setup.sh
    COMMAND ${CMAKE_SOURCE_DIR}/cmake_setup.sh ${CMAKE_SOURCE_DIR} build CUDA --module_name ${module_name}_cu --src ${pybind_cuda_SRCS}
)

add_custom_target(
        pytorch_build
        COMMAND chmod a+x ${CMAKE_SOURCE_DIR}/cmake_setup.sh
        COMMAND ${CMAKE_SOURCE_DIR}/cmake_setup.sh ${CMAKE_SOURCE_DIR} build --module_name ${module_name} --src ${pybind_cuda_SRCS}
)
