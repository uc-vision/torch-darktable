from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils import cpp_extension


def create_extension():
    """Create the PyTorch extension for PPG demosaic"""
    
    # Include directories
    include_dirs = cpp_extension.include_paths() + [pybind11.get_include()]
    
    # Source files - CUDA version
    sources = ["ppg_cuda.cpp", "ppg_kernels.cu"]
    
    # Compiler arguments - pure CUDA
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "--expt-relaxed-constexpr", "-arch=sm_70"]
    }
    
    # Create pure CUDA extension (no OpenCL dependencies)
    extension = cpp_extension.CUDAExtension(
        name="ppg_demosaic_cuda",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args
    )
    
    return extension

setup(
    name="ppg_demosaic",
    version="0.1.0",
    description="PPG Demosaic algorithm extracted from darktable for PyTorch",
    author="PPG Extractor",
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.7.0",
        "numpy>=2.2.0",
    ],
    ext_modules=[create_extension()],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    zip_safe=False,
)
