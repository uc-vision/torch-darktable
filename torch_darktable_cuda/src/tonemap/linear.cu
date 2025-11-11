#include "color_adaption.h"
#include "tonemap.h"
#include "device_math.h"
#include "cuda_utils.h"
#include "device_color_conversions.h"

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAStream.h>

// Linear tone mapping kernel  
__global__ void linear_tonemap_kernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    const float* __restrict__ metrics,
    float gamma,
    float vibrance,
    float light_adapt,
    float intensity,
    int height,
    int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float3 rgb = load<float3>(input, idx);
    
    // Compute adaptation from metrics and pixel values
    float3 adapt = compute_adaptation(metrics, rgb, light_adapt, intensity);

    float3 tonemapped = rgb / adapt; 
    float3 gamma_corrected = pow(max(tonemapped, 0.0f), 1.0f / gamma);
    float3 with_vibrance = modify_rgb_vibrance_dt(gamma_corrected, vibrance);

    store(clamp(with_vibrance, 0.0f, 1.0f), output, idx);
}


// Single struct-based implementation
torch::Tensor linear_tonemap(
    const torch::Tensor& image,
    const torch::Tensor& metrics,
    const TonemapParams& params) {

    check_image(image);
    TORCH_CHECK(metrics.dtype() == torch::kFloat32 && metrics.numel() == 5);

    int height = image.size(0);
    int width = image.size(1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto output = torch::empty({height, width, 3}, torch::dtype(torch::kUInt8).device(image.device()));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    linear_tonemap_kernel<<<grid_size, block_size, 0, stream>>>(
        image.data_ptr<float>(),
        output.data_ptr<unsigned char>(),
        metrics.data_ptr<float>(),
        params.gamma,
        params.vibrance,
        params.light_adapt,
        params.intensity,
        height,
        width
    );

    CUDA_CHECK_KERNEL();
    return output;
}
