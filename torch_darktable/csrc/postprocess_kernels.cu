/*
 * Post-Processing CUDA Kernels for Demosaicing
 * Color smoothing and green equilibration algorithms
 * Converted from OpenCL kernels for better PyTorch integration
 */

#include <cuda_runtime.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <algorithm>

#include "demosaic.h"

// FC inline function for Bayer pattern (from darktable)
__device__ __forceinline__ int FC(int row, int col, uint32_t filters) {
    return (filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1)) & 3;
}

// Compare and swap inline function for sorting network
__device__ __forceinline__ void cas(float& a, float& b) {
    float x = a;
    int c = a > b;
    a = c ? b : a;
    b = c ? x : b;
}

/**
 * Color smoothing using 3x3 median filter
 * Uses a sorting network to sort entirely in registers with no branches
 * Applied to R-G and B-G differences to preserve luminance
 */
__global__ void color_smoothing_kernel(
    float3* input,
    float3* output,
    int width,
    int height
) {
    extern __shared__ float3 smoothing_buffer[];
    
    const int lxid = threadIdx.x;
    const int lyid = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int lxsz = blockDim.x;
    const int buffwd = lxsz + 2;
    const int buffsz = (blockDim.x + 2) * (blockDim.y + 2);
    const int gsz = blockDim.x * blockDim.y;
    const int lidx = lyid * lxsz + lxid;

    const int nchunks = buffsz % gsz == 0 ? buffsz/gsz - 1 : buffsz/gsz;

    #pragma unroll
    for(int n = 0; n <= nchunks; n++)
    {
        const int bufidx = (n * gsz) + lidx;
        if(bufidx >= buffsz) break;

        // get position in buffer coordinates and from there translate to position in global coordinates
        const int gx = (bufidx % buffwd) - 1 + x - lxid;
        const int gy = (bufidx / buffwd) - 1 + y - lyid;

        // don't read more than needed
        if(gx >= width + 1 || gy >= height + 1) continue;

        smoothing_buffer[bufidx] = (gx >= 0 && gy >= 0 && gx < width && gy < height) ? 
                                   input[gy * width + gx] : make_float3(0.0f, 0.0f, 0.0f);
    }

    __syncthreads();

    if(x >= width || y >= height) return;

    // re-position buffer
    float3* centered_buffer = smoothing_buffer + (lyid + 1) * buffwd + lxid + 1;

    float3 o = centered_buffer[0];

    // 3x3 median for R (R-G difference)
    float s0 = centered_buffer[-buffwd - 1].x - centered_buffer[-buffwd - 1].y;
    float s1 = centered_buffer[-buffwd].x - centered_buffer[-buffwd].y;
    float s2 = centered_buffer[-buffwd + 1].x - centered_buffer[-buffwd + 1].y;
    float s3 = centered_buffer[-1].x - centered_buffer[-1].y;
    float s4 = centered_buffer[0].x - centered_buffer[0].y;
    float s5 = centered_buffer[1].x - centered_buffer[1].y;
    float s6 = centered_buffer[buffwd - 1].x - centered_buffer[buffwd - 1].y;
    float s7 = centered_buffer[buffwd].x - centered_buffer[buffwd].y;
    float s8 = centered_buffer[buffwd + 1].x - centered_buffer[buffwd + 1].y;

    cas(s1, s2);
    cas(s4, s5);
    cas(s7, s8);
    cas(s0, s1);
    cas(s3, s4);
    cas(s6, s7);
    cas(s1, s2);
    cas(s4, s5);
    cas(s7, s8);
    cas(s0, s3);
    cas(s5, s8);
    cas(s4, s7);
    cas(s3, s6);
    cas(s1, s4);
    cas(s2, s5);
    cas(s4, s7);
    cas(s4, s2);
    cas(s6, s4);
    cas(s4, s2);

    o.x = fmaxf(s4 + o.y, 0.0f);

    // 3x3 median for B (B-G difference)
    s0 = centered_buffer[-buffwd - 1].z - centered_buffer[-buffwd - 1].y;
    s1 = centered_buffer[-buffwd].z - centered_buffer[-buffwd].y;
    s2 = centered_buffer[-buffwd + 1].z - centered_buffer[-buffwd + 1].y;
    s3 = centered_buffer[-1].z - centered_buffer[-1].y;
    s4 = centered_buffer[0].z - centered_buffer[0].y;
    s5 = centered_buffer[1].z - centered_buffer[1].y;
    s6 = centered_buffer[buffwd - 1].z - centered_buffer[buffwd - 1].y;
    s7 = centered_buffer[buffwd].z - centered_buffer[buffwd].y;
    s8 = centered_buffer[buffwd + 1].z - centered_buffer[buffwd + 1].y;

    cas(s1, s2);
    cas(s4, s5);
    cas(s7, s8);
    cas(s0, s1);
    cas(s3, s4);
    cas(s6, s7);
    cas(s1, s2);
    cas(s4, s5);
    cas(s7, s8);
    cas(s0, s3);
    cas(s5, s8);
    cas(s4, s7);
    cas(s3, s6);
    cas(s1, s4);
    cas(s2, s5);
    cas(s4, s7);
    cas(s4, s2);
    cas(s6, s4);
    cas(s4, s2);

    o.z = fmaxf(s4 + o.y, 0.0f);

    output[y * width + x] = make_float3(fmaxf(o.x, 0.0f), fmaxf(o.y, 0.0f), fmaxf(o.z, 0.0f));
}

/**
 * Green equilibration - local averaging approach
 * Corrects green channel imbalance using local neighborhood analysis
 */
__global__ void green_eq_local_kernel(
    float3* input,
    float3* output,
    int width,
    int height,
    uint32_t filters,
    float threshold
) {
    extern __shared__ float green_eq_buffer[];
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int xlsz = blockDim.x;
    const int ylsz = blockDim.y;
    const int xlid = threadIdx.x;
    const int ylid = threadIdx.y;
    const int xgid = blockIdx.x;
    const int ygid = blockIdx.y;

    // individual control variable in this work group and the work group size
    const int l = ylid * xlsz + xlid;
    const int lsz = xlsz * ylsz;

    // stride and maximum capacity of local buffer
    // cells of 1*float per pixel with a surrounding border of 2 cells
    const int stride = xlsz + 2*2;
    const int maxbuf = stride * (ylsz + 2*2);

    // coordinates of top left pixel of buffer
    // this is 2 pixel left and above of the work group origin
    const int xul = xgid * xlsz - 2;
    const int yul = ygid * ylsz - 2;

    // populate local memory buffer
    #pragma unroll
    for(int n = 0; n <= maxbuf/lsz; n++)
    {
        const int bufidx = n * lsz + l;
        if(bufidx >= maxbuf) continue;
        const int xx = xul + bufidx % stride;
        const int yy = yul + bufidx / stride;
        green_eq_buffer[bufidx] = (xx >= 0 && yy >= 0 && xx < width && yy < height) ? 
                                  input[yy * width + xx].y : 0.0f; // Green channel only
    }

    // center buffer around current x,y-Pixel
    float* centered_buffer = green_eq_buffer + (ylid + 2) * stride + xlid + 2;

    __syncthreads();

    if(x >= width || y >= height) return;

    const int c = FC(y, x, filters);
    const float maximum = 1.0f;
    float3 pixel = input[y * width + x];
    float o = pixel.y; // Start with green channel

    if(c == 1 && (y & 1)) // Green2 pixels (odd rows)
    {
        const float o1_1 = centered_buffer[-1 * stride - 1];
        const float o1_2 = centered_buffer[-1 * stride + 1];
        const float o1_3 = centered_buffer[ 1 * stride - 1];
        const float o1_4 = centered_buffer[ 1 * stride + 1];
        const float o2_1 = centered_buffer[-2 * stride + 0];
        const float o2_2 = centered_buffer[ 2 * stride + 0];
        const float o2_3 = centered_buffer[-2];
        const float o2_4 = centered_buffer[ 2];

        const float m1 = (o1_1 + o1_2 + o1_3 + o1_4) / 4.0f;
        const float m2 = (o2_1 + o2_2 + o2_3 + o2_4) / 4.0f;

        if ((m2 > 0.0f) && (m1 > 0.0f) && (m1 / m2 < maximum * 2.0f))
        {
            const float c1 = (fabsf(o1_1 - o1_2) + fabsf(o1_1 - o1_3) + fabsf(o1_1 - o1_4) + 
                              fabsf(o1_2 - o1_3) + fabsf(o1_3 - o1_4) + fabsf(o1_2 - o1_4)) / 6.0f;
            const float c2 = (fabsf(o2_1 - o2_2) + fabsf(o2_1 - o2_3) + fabsf(o2_1 - o2_4) + 
                              fabsf(o2_2 - o2_3) + fabsf(o2_3 - o2_4) + fabsf(o2_2 - o2_4)) / 6.0f;

            if((o < maximum * 0.95f) && (c1 < maximum * threshold) && (c2 < maximum * threshold))
                o *= m1 / m2;
        }
    }

    pixel.y = fmaxf(o, 0.0f);
    output[y * width + x] = pixel;
}

/**
 * Green equilibration - global averaging first pass
 * Reduces green channel values across work groups
 */
__global__ void green_eq_global_reduce_first_kernel(
    float3* input,
    int width,
    int height,
    float2* accu,
    uint32_t filters
) {
    extern __shared__ float2 reduce_buffer[];
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int xlsz = blockDim.x;
    const int ylsz = blockDim.y;
    const int xlid = threadIdx.x;
    const int ylid = threadIdx.y;

    const int l = ylid * xlsz + xlid;

    const int c = FC(y, x, filters);

    const int isinimage = (x < 2 * (width / 2) && y < 2 * (height / 2));
    const int isgreen1 = (c == 1 && !(y & 1));
    const int isgreen2 = (c == 1 && (y & 1));

    float pixel = (x < width && y < height) ? input[y * width + x].y : 0.0f;

    reduce_buffer[l].x = isinimage && isgreen1 ? pixel : 0.0f;
    reduce_buffer[l].y = isinimage && isgreen2 ? pixel : 0.0f;

    __syncthreads();

    const int lsz = xlsz * ylsz;

    #pragma unroll
    for(int offset = lsz / 2; offset > 0; offset = offset / 2)
    {
        if(l < offset)
        {
            reduce_buffer[l] = make_float2(reduce_buffer[l].x + reduce_buffer[l + offset].x,
                                          reduce_buffer[l].y + reduce_buffer[l + offset].y);
        }
        __syncthreads();
    }

    const int xgid = blockIdx.x;
    const int ygid = blockIdx.y;
    const int xgsz = gridDim.x;

    const int m = ygid * xgsz + xgid;
    if(l == 0)
        accu[m] = reduce_buffer[0];
}



/**
 * Green equilibration - global averaging apply correction
 * Applies the calculated global ratio to Green1 pixels
 */
__global__ void green_eq_global_apply_kernel(
    float3* input,
    float3* output,
    int width,
    int height,
    uint32_t filters,
    float gr_ratio
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    float3 pixel = input[y * width + x];

    const int c = FC(y, x, filters);
    const int isgreen1 = (c == 1 && !(y & 1));

    pixel.y *= (isgreen1 ? gr_ratio : 1.0f);
    output[y * width + x] = make_float3(fmaxf(pixel.x, 0.0f), fmaxf(pixel.y, 0.0f), 
                                       fmaxf(pixel.z, 0.0f));
}




// (no packing kernel; all processing uses 3 channels natively)


// PostProcess Implementation
struct PostProcessImpl : public PostProcess {
    uint32_t filters_;
    int color_smoothing_passes_;
    bool green_eq_local_;
    bool green_eq_global_;
    float green_eq_threshold_;
    int width_;
    int height_;

    torch::Device device_;

    dim3 block_;
    dim3 grid_;

    // Working buffers for ping-pong operations
    torch::Tensor buffer1_;
    torch::Tensor buffer2_;
    torch::Tensor reduce_buffer_;

    

    PostProcessImpl(torch::Device device, int width, int height, uint32_t filters,
      int color_smoothing_passes,
      bool green_eq_local,
      bool green_eq_global,
      float green_eq_threshold)
        : device_(device), width_(width), height_(height), filters_(filters),
          color_smoothing_passes_(color_smoothing_passes),
          green_eq_local_(green_eq_local), green_eq_global_(green_eq_global),
          green_eq_threshold_(green_eq_threshold) {

        constexpr int block_size = 16;

        block_ = dim3(block_size, block_size);
        grid_ = dim3((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);

        // Initialize buffers with correct size from the start
        const auto buffer_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        buffer1_ = torch::empty({height, width, 3}, buffer_opts);
        buffer2_ = torch::empty({height, width, 3}, buffer_opts);
        reduce_buffer_ = torch::empty({grid_.x * grid_.y, 2}, buffer_opts);
    }

    ~PostProcessImpl() override = default;


    torch::Tensor process(const torch::Tensor& input) override {
        // Input validation
        TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
        TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
        TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (H, W, 3)");
        TORCH_CHECK(input.size(2) == 3, "Input must have 3 channels (RGB)");

        // Check dimensions match expected size
        TORCH_CHECK(input.size(0) == height_ && input.size(1) == width_, "Input size ", input.size(0), "x", input.size(1), " does not match expected ", height_, "x", width_);

        // Ensure input is contiguous and copy to buffer1
        auto contiguous_input = input.contiguous();
        buffer1_.copy_(contiguous_input);

        // Initialize local buffer pointers
        float3 *buffers[2] = {
            reinterpret_cast<float3*>(buffer1_.data_ptr<float>()),
            reinterpret_cast<float3*>(buffer2_.data_ptr<float>())
        };

        bool current = false;

        // Get CUDA stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // Use pre-calculated CUDA dimensions

        // Color smoothing (multiple passes using ping-pong buffers)
        const int smoothing_shared_size = (block_.x + 2) * (block_.y + 2) * sizeof(float3);

        for(int pass = 0; pass < color_smoothing_passes_; pass++) {
            color_smoothing_kernel<<<grid_, block_, smoothing_shared_size, stream>>>(
                buffers[current], buffers[!current],
                width_, height_);

            // Swap buffer pointers (no data copy)
            current = not current;
        }

        // Green equilibration - global
        if(green_eq_global_) {
            const int reduce_shared_size = block_.x * block_.y * sizeof(float2);

            // First reduction pass
            float2* reduce_ptr = reinterpret_cast<float2*>(reduce_buffer_.data_ptr<float>());

            green_eq_global_reduce_first_kernel<<<grid_, block_, reduce_shared_size, stream>>>(
                buffers[current], width_, height_,
                reduce_ptr, filters_);

            // Sum reduction using PyTorch
            auto final_result = reduce_buffer_.sum(0);

            float sum1 = final_result[0].item<float>();
            float sum2 = final_result[1].item<float>();
            float gr_ratio = (sum1 > 0.0f && sum2 > 0.0f) ? sum2 / sum1 : 1.0f;

            // Apply global correction
            green_eq_global_apply_kernel<<<grid_, block_, 0, stream>>>(
                buffers[current], buffers[!current],
                width_, height_, filters_, gr_ratio);

            // Swap buffer pointers (no data copy)
            current = not current;
        }

        // Green equilibration - local
        if(green_eq_local_) {
            const int green_eq_shared_size = (block_.x + 4) * (block_.y + 4) * sizeof(float);

            green_eq_local_kernel<<<grid_, block_, green_eq_shared_size, stream>>>(
                buffers[current], buffers[!current],
                width_, height_, filters_, green_eq_threshold_);

            // Swap buffer pointers (no data copy)
            current = not current;
        }

        return (current ? buffer2_ : buffer1_).clone();
    }
};

std::shared_ptr<PostProcess> create_postprocess(torch::Device device,
    int width, int height,
    uint32_t filters,
    int color_smoothing_passes,
    bool green_eq_local,
    bool green_eq_global,
    float green_eq_threshold) {
    return std::make_shared<PostProcessImpl>(device, width, height, filters,
      color_smoothing_passes,
      green_eq_local,
      green_eq_global,
      green_eq_threshold);
}
