#include <torch/extension.h>
#include <cuda_runtime.h>
#include "color_adaption.h"
#include "tonemap.h"

#include "device_math.h"
#include "cuda_utils.h"

#include <c10/cuda/CUDAStream.h>

__device__ __constant__ ColorTransform transform;
__device__ __constant__ float lower, upper;

// ACES tone mapping kernel  
__global__ void linear_tonemap_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float gamma,
    int height,
    int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float3 rgb = float3_load(input, idx);

    float3 scaled = (rgb - transform.bounds_min) / transform.range;
    float3 adapt = pow(transform.adapt_mean / transform.exposure, transform.map_key);

    float3 tonemapped = scaled / adapt; 
    float3_store(tonemapped, output, idx);
}

__global__ void apply_gamma_kernel(
    float* __restrict__ image,
    float gamma,
    int height,
    int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float3 rgb = (float3_load(image, idx) - lower) / (upper - lower);

    float3 gamma_corrected = pow(fmax(rgb, 0.0f), 1.0f / gamma);
    float3_store(gamma_corrected, image, idx);
}

// Single struct-based implementation
torch::Tensor linear_tonemap(
    const torch::Tensor& image,
    const torch::Tensor& metrics,
    const TonemapParams& params) {

    check_image(image);


    int height = image.size(0);
    int width = image.size(1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    metrics_to_transform(transform, metrics, params, stream);
    
    auto output = torch::empty({height, width, 3}, torch::dtype(torch::kFloat32).device(image.device()));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    linear_tonemap_kernel<<<grid_size, block_size, 0, stream>>>(
        image.data_ptr<float>(),
        output.data_ptr<float>(),
        params.gamma,
        height,
        width
    );

    auto bounds = compute_image_bounds({output}, 4);
    float* bounds_ptr = bounds.data_ptr<float>();

    CUDA_CHECK(cudaMemcpyToSymbolAsync(lower, bounds_ptr, sizeof(float), 0, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(upper, bounds_ptr + 1, sizeof(float), 0, cudaMemcpyDeviceToDevice, stream));
    
    apply_gamma_kernel<<<grid_size, block_size, 0, stream>>>(
        output.data_ptr<float>(),
        params.gamma,
        height,
        width
    );

    CUDA_CHECK_KERNEL();
    return output;
}
