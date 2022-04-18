//
// Created by Lazurite on 4/18/2022.
//

#include "cpp_extension.h"

void init_module_cpp_extension(pybind11::module &m){
    m.def("is_cpp_extension_init", [](){
        return true;
    });


}