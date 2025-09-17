#include "../cuda_utils.h"
#include "../device_math.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "color_adaption.h"
#include "tonemap.h"

#include <c10/cuda/CUDAStream.h>

__device__ __constant__ ColorTransform transform;

// ACES tone mapping kernel  
__global__ void linear_tonemap_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
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
    float3 gamma_corrected = pow(fmax(tonemapped, 0.0f), 1.0f / gamma);
    float3_to_uint8_rgb(gamma_corrected, output, idx);
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
    
    auto output = torch::empty({height, width, 3}, torch::dtype(torch::kUInt8).device(image.device()));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    linear_tonemap_kernel<<<grid_size, block_size, 0, stream>>>(
        image.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        params.gamma,
        height,
        width
    );

    CUDA_CHECK_KERNEL();
    return output;
}
