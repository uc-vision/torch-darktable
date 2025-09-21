// 5x5 linear demosaic using diamond window (13 taps) and per-pixel-type kernels
// Pattern selection via fc(filters) and array indexing (no control-flow branches on pattern)

#include <cuda_runtime.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#include "../cuda_utils.h"
#include "../device_math.h"
#include "demosaic.h"
#include "bayer_device.h"


// Diamond 5x5 offsets (13 taps) in Taichi's order: rows of 1,3,5,3,1
__device__ constexpr int2 offsets[13] = {
    {-2, 0},  
    {-1, -1}, {-1, 0}, {-1, 1},
    {0, -2}, {0, -1}, {0, 0}, {0, 1}, {0, 2},
    {1, -1}, {1, 0}, {1, 1},
    {2, 0}
};


// Four per-pixel-type kernels (R, G1, G2, B), each with 13 taps and 3 channels (R,G,B)
// Coefficients derived from taichi_image/bayer.py (symmetrical diamond weights)
__device__ constexpr float3 diamond_kernels[4][13] = {
    // Type 0: R pixel -> (ident, g_rb, rb_br)
    {
        {  0, -2, -3 },
        {  0,  0,  4 }, {  0,  4,  0 }, {  0,  0,  4 },
        {  0, -2, -3 }, {  0,  4,  0 }, { 16,  8, 12 }, {  0,  4,  0 }, {  0, -2, -3 },
        {  0,  0,  4 }, {  0,  4,  0 }, {  0,  0,  4 },
        {  0, -2, -3 }
    },
    // Type 1: G1 pixel -> (r_g1, ident, b_g1=r_g2)
    {
        { -2,  0,  1 },
        { -2,  0, -2 }, {  8,  0,  0 }, { -2,  0, -2 },
        {  1,  0, -2 }, {  0,  0,  8 }, { 10, 16, 10 }, {  0,  0,  8 }, {  1,  0, -2 },
        { -2,  0, -2 }, {  8,  0,  0 }, { -2,  0, -2 },
        { -2,  0,  1 }
    },
    // Type 2: G2 pixel -> (r_g2, ident, b_g2=r_g1)
    {
        {  1,  0, -2 },
        { -2,  0, -2 }, {  0,  0,  8 }, { -2,  0, -2 },
        { -2,  0,  1 }, {  8,  0,  0 }, { 10, 16, 10 }, {  8,  0,  0 }, { -2,  0,  1 },
        { -2,  0, -2 }, {  0,  0,  8 }, { -2,  0, -2 },
        {  1,  0, -2 }
    },
    // Type 3: B pixel -> (rb_br, g_rb, ident)
    {
        { -3, -2,  0 },
        {  4,  0,  0 }, {  0,  4,  0 }, {  4,  0,  0 },
        { -3, -2,  0 }, {  0,  4,  0 }, { 12,  8, 16 }, {  0,  4,  0 }, { -3, -2,  0 },
        {  4,  0,  0 }, {  0,  4,  0 }, {  4,  0,  0 },
        { -3, -2,  0 }
    }
};


template<BayerPattern pattern>
__global__ void bilinear5x5(
    const float* __restrict__ input, // HxW mono Bayer
    float3* __restrict__ output,     // HxW float3
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;


    #pragma unroll
    for (int C = 0; C < 4; ++C) {

      const int pixel_type = get_pixel_type<pattern>(C);
      int2 p = make_int2(x, y) * 2 + offset2x2(C);
      if (p.x >= width || p.y >= height) continue;

      float3 acc_r = make_float3(0.0f, 0.0f, 0.0f);
      float3 sum_r = make_float3(0.0f, 0.0f, 0.0f);

      #pragma unroll
      for (int k = 0; k < 13; ++k) {
          const int2 d = offsets[k];

          const int2 c = p + d;
          const float v = input[clamp(c.y, 0, height - 1) * width + clamp(c.x, 0, width - 1)];

          const float3 w = diamond_kernels[pixel_type][k];
          acc_r += w * v;
          sum_r += w;
      }

      output[p.y * width + p.x] = acc_r / sum_r;
    }
}



// Host entry to run 5x5 linear demosaic
torch::Tensor bilinear5x5_demosaic(const torch::Tensor& input, BayerPattern pattern) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (H, W, 1)");
    TORCH_CHECK(input.size(2) == 1, "Input must have single channel (raw Bayer)");

    const int height = input.size(0);
    const int width = input.size(1);

    auto contiguous_input = input.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const auto out_opts = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor output = torch::empty({height, width, 3}, out_opts);

    dim3 block = block_size_2d;
    dim3 grid = grid2d((width + 1) / 2, (height + 1) / 2);


    float* input_ptr = contiguous_input.data_ptr<float>();
    float3* output_ptr = reinterpret_cast<float3*>(output.data_ptr<float>());

    switch (pattern) {
       case BayerPattern::RGGB:
          bilinear5x5<BayerPattern::RGGB><<<grid, block, 0, stream>>>(
              input_ptr, output_ptr, width, height);
          break;
        case BayerPattern::BGGR:
          bilinear5x5<BayerPattern::BGGR><<<grid, block, 0, stream>>>(
              input_ptr, output_ptr, width, height);
          break;
        case BayerPattern::GRBG:
          bilinear5x5<BayerPattern::GRBG><<<grid, block, 0, stream>>>(
              input_ptr, output_ptr, width, height);
          break;
        case BayerPattern::GBRG:
          bilinear5x5<BayerPattern::GBRG><<<grid, block, 0, stream>>>(
              input_ptr, output_ptr, width, height);
          break;
        default:
          TORCH_CHECK(false, "Invalid pattern");
    }

    return output;
}


