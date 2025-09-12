#include "cuda_utils.h"
#include "device_math.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Encode two 12-bit values into three 8-bit bytes (standard format)
__device__ __forceinline__ void encode12_pair(uint16_t p0, uint16_t p1, uint8_t* output) {
    output[0] = p0 & 0xff;
    output[1] = ((p1 & 0xf) << 4) | (p0 >> 8);
    output[2] = p1 >> 4;
}

// Decode three 8-bit bytes into two 12-bit values (standard format)
__device__ __forceinline__ void decode12_pair(const uint8_t* input, uint16_t* p0, uint16_t* p1) {
    *p0 = ((uint16_t(input[1]) & 0xf) << 8) | uint16_t(input[0]);
    *p1 = (uint16_t(input[2]) << 4) | (uint16_t(input[1]) >> 4);
}

// Encode two 12-bit values into three 8-bit bytes (IDS format)
__device__ __forceinline__ void encode12_pair_ids(uint16_t p0, uint16_t p1, uint8_t* output) {
    output[0] = p0 >> 4;
    output[1] = p1 >> 4;
    output[2] = ((p0 & 0xf) << 4) | (p1 & 0xf);
}

// Decode three 8-bit bytes into two 12-bit values (IDS format)
__device__ __forceinline__ void decode12_pair_ids(const uint8_t* input, uint16_t* p0, uint16_t* p1) {
    *p0 = (uint16_t(input[0]) << 4) | (uint16_t(input[2]) & 0xf);
    *p1 = (uint16_t(input[1]) << 4) | (uint16_t(input[2]) >> 4);
}

// Encode kernel for uint16 to packed 12-bit
__global__ void encode12_kernel_u16(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int num_pairs,
    bool ids_format
) {
    int idx = thread_index();
    if (idx >= num_pairs) return;

    uint16_t p0 = input[idx * 2];
    uint16_t p1 = input[idx * 2 + 1];
    
    // Clamp to 12-bit range
    p0 = min(p0, uint16_t(4095));
    p1 = min(p1, uint16_t(4095));

    uint8_t* out_ptr = output + idx * 3;
    
    if (ids_format) {
        encode12_pair_ids(p0, p1, out_ptr);
    } else {
        encode12_pair(p0, p1, out_ptr);
    }
}

// Encode kernel for float with scaling
__global__ void encode12_kernel_float(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    int num_pairs,
    bool ids_format,
    float scale
) {
    int idx = thread_index();
    if (idx >= num_pairs) return;

    float f0 = input[idx * 2] * scale;
    float f1 = input[idx * 2 + 1] * scale;
    
    // Round and clamp to 12-bit range
    uint16_t p0 = min(uint16_t(roundf(f0)), uint16_t(4095));
    uint16_t p1 = min(uint16_t(roundf(f1)), uint16_t(4095));

    uint8_t* out_ptr = output + idx * 3;
    
    if (ids_format) {
        encode12_pair_ids(p0, p1, out_ptr);
    } else {
        encode12_pair(p0, p1, out_ptr);
    }
}

// Decode kernel to float
__global__ void decode12_kernel_float(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int num_pairs,
    bool ids_format,
    float scale
) {
    int idx = thread_index();
    if (idx >= num_pairs) return;

    const uint8_t* in_ptr = input + idx * 3;
    uint16_t p0, p1;
    
    if (ids_format) {
        decode12_pair_ids(in_ptr, &p0, &p1);
    } else {
        decode12_pair(in_ptr, &p0, &p1);
    }
    
    output[idx * 2] = float(p0) * scale;
    output[idx * 2 + 1] = float(p1) * scale;
}

// Decode kernel to half precision
__global__ void decode12_kernel_half(
    const uint8_t* __restrict__ input,
    __half* __restrict__ output,
    int num_pairs,
    bool ids_format,
    float scale
) {
    int idx = thread_index();
    if (idx >= num_pairs) return;

    const uint8_t* in_ptr = input + idx * 3;
    uint16_t p0, p1;
    
    if (ids_format) {
        decode12_pair_ids(in_ptr, &p0, &p1);
    } else {
        decode12_pair(in_ptr, &p0, &p1);
    }
    
    output[idx * 2] = __float2half(float(p0) * scale);
    output[idx * 2 + 1] = __float2half(float(p1) * scale);
}

// Decode kernel to uint16
__global__ void decode12_kernel_u16(
    const uint8_t* __restrict__ input,
    uint16_t* __restrict__ output,
    int num_pairs,
    bool ids_format
) {
    int idx = thread_index();
    if (idx >= num_pairs) return;

    const uint8_t* in_ptr = input + idx * 3;
    uint16_t p0, p1;
    
    if (ids_format) {
        decode12_pair_ids(in_ptr, &p0, &p1);
    } else {
        decode12_pair(in_ptr, &p0, &p1);
    }
    
    output[idx * 2] = p0;
    output[idx * 2 + 1] = p1;
}

// Host functions
torch::Tensor encode12_u16(const torch::Tensor& input, bool ids_format) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kUInt16, "Input must be uint16");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D tensor");
    TORCH_CHECK(input.size(0) % 2 == 0, "Input length must be even");

    int num_pairs = input.size(0) / 2;
    auto output = torch::empty({num_pairs * 3}, torch::dtype(torch::kUInt8).device(input.device()));

    const int block_size = 256;
    const int num_blocks = (num_pairs + block_size - 1) / block_size;

    encode12_kernel_u16<<<num_blocks, block_size>>>(
        input.data_ptr<uint16_t>(),
        output.data_ptr<uint8_t>(),
        num_pairs,
        ids_format
    );

    CUDA_CHECK_KERNEL();
    return output;
}

torch::Tensor encode12_float(const torch::Tensor& input, bool ids_format, bool scaled) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D tensor");
    TORCH_CHECK(input.size(0) % 2 == 0, "Input length must be even");

    int num_pairs = input.size(0) / 2;
    auto output = torch::empty({num_pairs * 3}, torch::dtype(torch::kUInt8).device(input.device()));

    float scale = scaled ? 4095.0f : 1.0f;

    const int block_size = 256;
    const int num_blocks = (num_pairs + block_size - 1) / block_size;

    encode12_kernel_float<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        num_pairs,
        ids_format,
        scale
    );

    CUDA_CHECK_KERNEL();
    return output;
}

torch::Tensor decode12_float(const torch::Tensor& input, bool ids_format, bool scaled) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D tensor");
    TORCH_CHECK(input.size(0) % 3 == 0, "Input length must be multiple of 3");

    int num_pairs = input.size(0) / 3;
    auto output = torch::empty({num_pairs * 2}, torch::dtype(torch::kFloat32).device(input.device()));

    float scale = scaled ? (1.0f / 4095.0f) : 1.0f;

    const int block_size = 256;
    const int num_blocks = (num_pairs + block_size - 1) / block_size;

    decode12_kernel_float<<<num_blocks, block_size>>>(
        input.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        num_pairs,
        ids_format,
        scale
    );

    CUDA_CHECK_KERNEL();
    return output;
}

torch::Tensor decode12_half(const torch::Tensor& input, bool ids_format, bool scaled) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D tensor");
    TORCH_CHECK(input.size(0) % 3 == 0, "Input length must be multiple of 3");

    int num_pairs = input.size(0) / 3;
    auto output = torch::empty({num_pairs * 2}, torch::dtype(torch::kFloat16).device(input.device()));

    float scale = scaled ? (1.0f / 4095.0f) : 1.0f;

    const int block_size = 256;
    const int num_blocks = (num_pairs + block_size - 1) / block_size;

    decode12_kernel_half<<<num_blocks, block_size>>>(
        input.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        num_pairs,
        ids_format,
        scale
    );

    CUDA_CHECK_KERNEL();
    return output;
}

torch::Tensor decode12_u16(const torch::Tensor& input, bool ids_format) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D tensor");
    TORCH_CHECK(input.size(0) % 3 == 0, "Input length must be multiple of 3");

    int num_pairs = input.size(0) / 3;
    auto output = torch::empty({num_pairs * 2}, torch::dtype(torch::kUInt16).device(input.device()));

    const int block_size = 256;
    const int num_blocks = (num_pairs + block_size - 1) / block_size;

    decode12_kernel_u16<<<num_blocks, block_size>>>(
        input.data_ptr<uint8_t>(),
        output.data_ptr<uint16_t>(),
        num_pairs,
        ids_format
    );

    CUDA_CHECK_KERNEL();
    return output;
}
