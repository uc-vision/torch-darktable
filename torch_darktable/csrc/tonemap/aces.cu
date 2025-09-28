#include "../cuda_utils.h"
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "color_adaption.h"
#include "tonemap.h"
#include "device_math.h"
#include "device_color_conversions.h"

#include <c10/cuda/CUDAStream.h>


__device__ __forceinline__ float3 RRTAndODTFit(float3 v) {
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

__device__ __forceinline__ float3 aces_tonemap(const float3 rgb) {
  const float3x3 aces_input = float3x3(
    0.59719f, 0.35458f, 0.04823f,
    0.07600f, 0.90834f, 0.01566f,
    0.02840f, 0.13383f, 0.83777f);

  const float3x3 aces_output = float3x3(
    1.60475f, -0.53108f, -0.07367f,
    -0.10208f,  1.10813f, -0.00605f,
    -0.00327f, -0.07276f,  1.07602f);


    float3 aces_in = aces_input * rgb;
    float3 compressed = RRTAndODTFit(aces_in);
    return aces_output * compressed;
}

// ACES tone mapping kernel  
__global__ void aces_adaptive_kernel(
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
    float3 adjustment = compute_adaptation(metrics, rgb, light_adapt, intensity);
    float3 tonemapped = aces_tonemap(rgb / adjustment);
    
    float3 gamma_corrected = pow(max(tonemapped, 0.0f), 1.0f / gamma);
    float3 with_vibrance = modify_rgb_vibrance_dt(gamma_corrected, vibrance);
    store(with_vibrance, output, idx);
}


// ACES tone mapping kernel  
__global__ void aces_tonemap_kernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    float intensity,
    float gamma,
    float vibrance,
    int height,
    int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float3 rgb = load<float3>(input, idx);

    float3 tonemapped = aces_tonemap(rgb * pow(2.0f, intensity));

    float3 gamma_corrected = pow(max(tonemapped, 0.0f), 1.0f / gamma);
    float3 with_vibrance = modify_rgb_vibrance_dt(gamma_corrected, vibrance);
    store(with_vibrance, output, idx);
}


torch::Tensor aces_tonemap(
    const torch::Tensor& image,
    const TonemapParams& params
) {
    check_image(image);

    int height = image.size(0);
    int width = image.size(1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto output = torch::empty({height, width, 3}, torch::dtype(torch::kUInt8).device(image.device()));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    aces_tonemap_kernel<<<grid_size, block_size, 0, stream>>>(
        image.data_ptr<float>(),
        output.data_ptr<unsigned char>(),
        params.intensity,
        params.gamma,
        params.vibrance,
        height,
        width
    );
    
    CUDA_CHECK_KERNEL();
    return output;
}

torch::Tensor adaptive_aces_tonemap(
    const torch::Tensor& image,
    const torch::Tensor& metrics,
    const TonemapParams& params
) {
    check_image(image);
    TORCH_CHECK(metrics.dtype() == torch::kFloat32 && metrics.numel() == 5);

    int height = image.size(0);
    int width = image.size(1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto output = torch::empty({height, width, 3}, torch::dtype(torch::kUInt8).device(image.device()));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    aces_adaptive_kernel<<<grid_size, block_size, 0, stream>>>(
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
