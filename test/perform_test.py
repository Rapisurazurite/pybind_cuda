from cpp_extension_test import python_implementation, cpp_implementation, python_implementation_cu, cpp_implementation_cu
from cuda_extension_test import cu_implementation

if __name__ == '__main__':

    python_implementation()
    cpp_implementation()

    python_implementation_cu()
    cpp_implementation_cu()

    cu_implementation()

    # CppExtension
    # Forward: 103.458 us | Backward 152.045 us
    # Forward: 92.468 us | Backward 254.263 us
    # Forward: 343.601 us | Backward 550.167 us
    # Forward: 310.490 us | Backward 932.610 us
    # Forward: 290.819 us | Backward 527.641 us

    # CUDAExtension
    # Forward: 115.890 us | Backward 171.866 us
    # Forward: 100.845 us | Backward 276.542 us
    # Forward: 347.710 us | Backward 560.175 us
    # Forward: 314.089 us | Backward 944.910 us
    # Forward: 291.789 us | Backward 528.431 us