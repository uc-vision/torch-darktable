cmake_minimum_required(VERSION 3.20...3.27)
project(torch_darktable_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Use shared CUDA runtime instead of static
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Find Python
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Source files
set(CUDA_SOURCES
    src/extension.cpp
    src/debayer/bilinear.cu
    src/debayer/ppg.cu
    src/debayer/rcd.cu
    src/debayer/postprocess.cu
    src/local_contrast/laplacian.cu
    src/local_contrast/bilateral.cu
    src/color_conversions.cu
    src/packed.cu
    src/tonemap/color_adaption.cu
    src/tonemap/aces.cu
    src/tonemap/linear.cu
    src/tonemap/reinhard.cu
    src/white_balance.cu
    src/denoise/denoise.cu
    src/jpeg_encoder.cu
)

# Create the extension module
pybind11_add_module(torch_darktable_extension ${CUDA_SOURCES})

# Set include directories
target_include_directories(torch_darktable_extension PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(torch_darktable_extension PRIVATE
    torch
    torch_python
    c10
    CUDA::cudart
    CUDA::nvjpeg
)

# Set compilation flags
target_compile_options(torch_darktable_extension PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wno-stringop-overread>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)


# Set CUDA architectures (adjust as needed)
# 86 = RTX 30xx, 89 = RTX 40xx, 90 = H100, 120 = 50xx
set_property(TARGET torch_darktable_extension PROPERTY CUDA_ARCHITECTURES 86 89)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Install to Python site-packages
if(DEFINED ENV{SP_DIR})
  install(TARGETS torch_darktable_extension LIBRARY DESTINATION $ENV{SP_DIR})
else()
  install(TARGETS torch_darktable_extension LIBRARY DESTINATION ${Python_SITEARCH})
endif()

