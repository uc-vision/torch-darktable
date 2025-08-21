/*
 * PPG Demosaic CUDA/PyTorch Extension
 * Pure CUDA implementation - no OpenCL interop needed
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// Forward declare CUDA kernel launchers
extern "C" {
    void launch_pre_median(float* input, float* output, int width, int height, 
                          uint32_t filters, float threshold, cudaStream_t stream);
    void launch_ppg_green(float* input, float* output, int width, int height, 
                         uint32_t filters, cudaStream_t stream);
    void launch_ppg_redblue(float* green_input, float* raw_input, float4* output, 
                           int width, int height, uint32_t filters, cudaStream_t stream);
    void launch_border_interpolate(float4* data, int width, int height, 
                                  uint32_t filters, int border, cudaStream_t stream);
}

torch::Tensor ppg_demosaic_cuda(torch::Tensor input, uint32_t filters, float median_threshold = 0.0f) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (H, W, 1)");
    TORCH_CHECK(input.size(2) == 1, "Input must have single channel (raw Bayer)");
    
    // Ensure input is contiguous
    input = input.contiguous();
    
    const int height = input.size(0);
    const int width = input.size(1);
    
    // Create output tensor (H, W, 4) for RGBA
    auto output = torch::zeros({height, width, 4}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Get raw pointers
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Create temporary buffers for intermediate steps
    auto temp_green = torch::zeros({height, width}, 
                                  torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    float* temp_green_ptr = temp_green.data_ptr<float>();
    
    auto temp_median = torch::zeros({height, width}, 
                                   torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    float* temp_median_ptr = temp_median.data_ptr<float>();
    
    // Step 1: Pre-median filtering (optional)
    if (median_threshold > 0.0f) {
        launch_pre_median(input_ptr, temp_median_ptr, width, height, filters, median_threshold, stream);
        // Swap to use median-filtered data
        float* temp = input_ptr;
        input_ptr = temp_median_ptr;
    }
    
    // Step 2: Green channel interpolation
    launch_ppg_green(input_ptr, temp_green_ptr, width, height, filters, stream);
    
    // Step 3: Red/Blue interpolation 
    // Cast output pointer to float4* for kernel
    float4* output_rgba = reinterpret_cast<float4*>(output_ptr);
    launch_ppg_redblue(temp_green_ptr, input_ptr, output_rgba, width, height, filters, stream);
    
    // Step 4: Border interpolation
    launch_border_interpolate(output_rgba, width, height, filters, 1, stream);
    
    // Synchronize to ensure completion
    cudaStreamSynchronize(stream);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ppg_demosaic", &ppg_demosaic_cuda, "PPG Demosaic (Pure CUDA)",
          py::arg("input"), py::arg("filters"), py::arg("median_threshold") = 0.0f);
}