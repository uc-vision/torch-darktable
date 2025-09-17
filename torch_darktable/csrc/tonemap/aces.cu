#include "../cuda_utils.h"
#include "../device_math.h"
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "color_adaption.h"
#include "tonemap.h"

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

__device__ __constant__ ColorTransform transform;

// ACES tone mapping kernel  
__global__ void aces_tonemap_kernel(
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

    float3 adjustment = pow(transform.adapt_mean / transform.exposure, transform.map_key);
    float3 tonemapped = aces_tonemap(scaled / adjustment);

    float3 gamma_corrected = pow(fmax(tonemapped, 0.0f), 1.0f / gamma);
    float3_to_uint8_rgb(gamma_corrected, output, idx);
}



torch::Tensor aces_tonemap(
    const torch::Tensor& image,
    const torch::Tensor& metrics,
    const TonemapParams& params
) {
    check_image(image);

    int height = image.size(0);
    int width = image.size(1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    metrics_to_transform(transform, metrics, params, stream);
    
    auto output = torch::empty({height, width, 3}, torch::dtype(torch::kUInt8).device(image.device()));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    aces_tonemap_kernel<<<grid_size, block_size>>>(
        image.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        params.gamma,
        height,
        width
    );

    CUDA_CHECK_KERNEL();
    return output;
}
