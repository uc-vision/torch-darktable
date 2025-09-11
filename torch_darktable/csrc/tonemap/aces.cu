#include "../cuda_utils.h"
#include "../device_math.h"
#include <torch/extension.h>
#include <cuda_runtime.h>

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
    
    // ACES input matrix
    const float3x3 aces_input(
        0.59719f, 0.35458f, 0.04823f,
        0.07600f, 0.90834f, 0.01566f,
        0.02840f, 0.13383f, 0.83777f
    );
    
    // ACES output matrix
    const float3x3 aces_output(
         1.60475f, -0.53108f, -0.07367f,
        -0.10208f,  1.10813f, -0.00605f,
        -0.00327f, -0.07276f,  1.07602f
    );
    
    int idx = y * width + x;
    float3 rgb = float3_load(input, idx);
    
    // Apply ACES input transform
    float3 aces_in = aces_input * rgb;
    
    // RRT and ODT fit
    float3 a = aces_in * (aces_in + 0.0245786f) - 0.000090537f;
    float3 b = aces_in * (0.983729f * aces_in + 0.432951f) + 0.238081f;
    float3 tonemapped = a / b;
    
    // Apply ACES output transform
    float3 aces_out = aces_output * tonemapped;
    
    // Apply gamma correction and convert to 8-bit
    float3 gamma_corrected = pow(fmax(aces_out, 0.0f), 1.0f / gamma);
    float3_to_uint8_rgb(gamma_corrected, output, idx);
}

torch::Tensor aces_tonemap(const torch::Tensor& image, float gamma) {
    TORCH_CHECK(image.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(image.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "Input must be (H, W, 3)");

    int height = image.size(0);
    int width = image.size(1);
    
    auto output = torch::empty({height, width, 3}, torch::dtype(torch::kUInt8).device(image.device()));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    aces_tonemap_kernel<<<grid_size, block_size>>>(
        image.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        gamma,
        height,
        width
    );

    CUDA_CHECK_KERNEL();
    return output;
}
