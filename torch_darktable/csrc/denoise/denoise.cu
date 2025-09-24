#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "reduction.h"
#include "denoise.h"
// #include "fft/fft_coop.h"
#include "fft/common.h"
#include "window.h"
#include "cuda_utils.h"
#include "pixel.h"

#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

namespace {

constexpr float eps = 1e-15f;

// Fixed-size constant memory for noise sigmas (max 3 channels)
__constant__ float noise_sigmas[3];

// Helper to upload noise sigmas to constant memory
template<typename T>
inline void upload_noise_sigmas(const torch::Tensor& noise_sigmas_tensor, cudaStream_t stream) {
    constexpr int C = channels<T>();
    CUDA_CHECK(cudaMemcpyToSymbolAsync(noise_sigmas, noise_sigmas_tensor.data_ptr<float>(), 
                                      C * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream));
}


__device__ __forceinline__ float block_mean(float value) {
    auto block = cg::this_thread_block();
    __shared__ float sum;
    
    if (block.thread_rank() == 0) {
        sum = 0.0f;
    }
    
    auto group = cg::coalesced_threads();
    float result = cg::reduce(group, value, cg::plus<float>{});
    if (group.thread_rank() == 0) {
        atomicAdd(&sum, result);
    }
    block.sync();
    
    return sum / block.size();
}

__device__ __forceinline__ float3 block_mean(float3 value) {
    auto block = cg::this_thread_block();
    __shared__ float3 sum;
    
    if (block.thread_rank() == 0) {
        sum = make_float3(0.0f, 0.0f, 0.0f);
    }
    
    auto group = cg::coalesced_threads();
    float3 result = cg::reduce(group, value, cg::plus<float3>{});
    if (group.thread_rank() == 0) {
        atomic_add(&sum, result);
    }
    block.sync();
    
    return sum / block.size();
}

template<int K, typename T>
__device__ __forceinline__ T compute_tile_mean(T row_data[K]) {
    // Each thread sums its row
    T row_sum = {};
    #pragma unroll
    for (int col = 0; col < K; col++) {
        row_sum = row_sum + row_data[col];
    }
    
    // Use block_mean to average across all threads (all rows)
    T tile_mean = block_mean(row_sum);
    
    // Normalize by total pixels in tile (K*K)
    return tile_mean / (K * K);
}


// Using custom FFT implementation

__device__ __forceinline__ int2 get_group_pos() {
    auto block = cg::this_thread_block();
    dim3 g = block.group_index();
    return make_int2(g.x, g.y);
}

__device__ __forceinline__ int reflect_1d(int x, int limit) {
    if (x < 0) x = -x;
    if (x >= limit) x = 2 * limit - x - 1;
    return x;
}

template<int K, typename T>
__device__ __forceinline__ void load_row(
    const T* __restrict__ img,
    int stride, int H, int W,
    int row, T row_data[K]
) {
    int2 g = get_group_pos();
    int2 grid_offset = {K / stride, K / stride};
    
    // Row-constant calculations
    int src_y = (g.y - grid_offset.y) * stride + row;
    int refl_y = reflect_1d(src_y, H);
    int base_y_offset = refl_y * W;
    
    #pragma unroll
    for (int col = 0; col < K; col++) {
        int src_x = (g.x - grid_offset.x) * stride + col;
        int refl_x = reflect_1d(src_x, W);
        row_data[col] = img[base_y_offset + refl_x];
    }
}

template<int K, typename T>
__device__ __forceinline__ void store_row(
    T* __restrict__ out,
    float* __restrict__ mask,
    int stride, int H_pad, int W_pad,
    int row, const T row_data[K],
    const T& mean
) {
    int2 g = get_group_pos();
    int2 grid_offset = {K / stride, K / stride};
    
    // Row-constant calculations
    int out_y = (g.y - grid_offset.y) * stride + row + K;    
    if (out_y >= H_pad) return;

    int base_y_offset = out_y * W_pad;
    int base_out_x = (g.x - grid_offset.x) * stride + K;
    
    #pragma unroll
    for (int col = 0; col < K; col++) {
        int out_x = base_out_x + col;
        
        if (out_x < W_pad) {
            int2 pos = make_int2(col, row);
            float fft_window = Window<K>::fft_window(pos);
            auto interp_window = Window<K>::interp_window(pos);
            
            int out_idx = base_y_offset + out_x;
            T reconstructed = (row_data[col] + mean * fft_window) * interp_window;
            
            atomic_add(&out[out_idx], reconstructed);
            atomicAdd(&mask[out_idx], fft_window * interp_window);
        }
    }
}


__device__ __forceinline__ Complex apply_gain(Complex value, float sigma) {
  float power = value.magnitude_squared() + eps;
  float gain = fmaxf(power - sigma * sigma, 0.0f) / power;
  return gain * value;
}



// Main kernel: orchestrates the Wiener filtering pipeline
template<int K, typename T>
__global__ void wiener_tile_kernel(
    const T* __restrict__ img,
    T* __restrict__ out,
    float* __restrict__ mask,
    int H, int W, int H_pad, int W_pad,
    int stride
) {
    int row = threadIdx.x;  // Thread processes row 'row' (0 to K-1)
    
    // Each thread loads one row (K pixels)
    T row_data[K];
    load_row<K, T>(img, stride, H, W, row, row_data);
    
    // Compute mean across entire tile using all threads
    T tile_mean = compute_tile_mean<K, T>(row_data);
    
    // Apply windowing and subtract mean
    #pragma unroll
    for (int col = 0; col < K; col++) {
        int2 pos = make_int2(col, row);
        float fft_window = Window<K>::fft_window(pos);
        row_data[col] = (row_data[col] - tile_mean) * fft_window;
    }
    
    // TODO: FFT processing will go here
    
    // Store row back to output
    store_row<K, T>(out, mask, stride, H_pad, W_pad, row, row_data, tile_mean);
}


template<int K, typename T>
__global__ void normalize_and_crop_kernel(
    const T* __restrict__ padded_out,  // (C, H_pad, W_pad)
    T* __restrict__ final_out,         // (C, H, W) - final output
    const float* __restrict__ mask,        // (H_pad, W_pad)

    int H, int W, int H_pad, int W_pad
) {
  
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, 
                         blockIdx.y * blockDim.y + threadIdx.y);
    
    if (pos.y < H && pos.x < W) {
        int2 pad_pos = pos + K;
        int padded_idx = pad_pos.y * W_pad + pad_pos.x;
        int idx = (pos.y * W + pos.x);  // HWC layout

        final_out[idx] = padded_out[padded_idx] / (mask[padded_idx] + eps);
    }
}


template<int K>
struct WienerImpl final : public Wiener {
    static_assert(K == 16 || K == 32, "K must be 16 or 32");

    torch::Device device_;
    int overlap_factor_;

public:
    WienerImpl(torch::Device device,
      int overlap_factor = 4,  
      const float interp_scale = 0.3f, const float fft_scale = 0.3f)
        : device_(device), overlap_factor_(overlap_factor) {
        
        auto stream = at::cuda::getCurrentCUDAStream(device.index());
        Window<K>::init(fft_scale, interp_scale, stream);
    }
    
private:
    template<int C>
    torch::Tensor _process(const torch::Tensor &input, const torch::Tensor &noise_sigmas) {
        static_assert(C == 1 || C == 3, "C must be 1 or 3");
        
        using pixel = std::conditional_t<C == 1, float, float3>;
        TORCH_CHECK(input.device() == device_, "input device mismatch");
        TORCH_CHECK(input.dim() == 3, "expected HWC tensor");

        const int H = static_cast<int>(input.size(0));
        const int W = static_cast<int>(input.size(1));
        const int input_C = static_cast<int>(input.size(2));
        TORCH_CHECK(input_C == C, "input channel count mismatch");
        
        // Compute dimensions on-demand
        const int stride = K / overlap_factor_;  // Configurable overlap
        const int h_pad = H + 2 * K;
        const int w_pad = W + 2 * K;
        
        // Grid needs to start earlier to cover boundaries with proper overlap
        const int grid_start = -(K / stride);  // Start early for boundary coverage
        const int grid_h = (H + K + stride - 1) / stride - grid_start;  // Extended coverage
        const int grid_w = (W + K + stride - 1) / stride - grid_start;
        
        auto padded_out = torch::zeros({h_pad, w_pad, C}, input.options());
        auto mask = torch::zeros({h_pad, w_pad}, input.options());
        auto final_out = torch::empty({H, W, C}, input.options());
        
        // Process noise sigmas
        TORCH_CHECK(noise_sigmas.numel() == C, "noise_sigmas must have C elements");
        
        dim3 grid(grid_w, grid_h);
        dim3 block(K);  // K threads, each handles one row of K pixels

        auto stream = at::cuda::getCurrentCUDAStream(device_.index());
        
        // Upload noise sigmas to constant memory
        upload_noise_sigmas<pixel>(noise_sigmas, stream);
       
        wiener_tile_kernel<K, pixel><<<grid, block, 0, stream>>>(
            reinterpret_cast<pixel const*>(input.data_ptr<float>()),
            reinterpret_cast<pixel*>(padded_out.data_ptr<float>()),
            mask.data_ptr<float>(),
            H, W, h_pad, w_pad,
            stride
        );

        
        // Check for kernel launch errors
        auto cuda_err = cudaGetLastError();
        TORCH_CHECK(cuda_err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(cuda_err));
        
        // Normalize and crop in a single kernel
        dim3 norm_grid(div_up(W, K), div_up(H, K));
        dim3 norm_block(K, K);
        
        normalize_and_crop_kernel<K, pixel><<<norm_grid, norm_block, 0, stream>>>(
            reinterpret_cast<pixel const*>(padded_out.data_ptr<float>()),
            reinterpret_cast<pixel*>(final_out.data_ptr<float>()),
            mask.data_ptr<float>(),
            H, W, h_pad, w_pad
        );
        
        CUDA_CHECK_KERNEL();
        
        return final_out;
    }

public:
    torch::Tensor process(const torch::Tensor &input, const torch::Tensor &noise_sigmas) override {
        TORCH_CHECK(input.dim() == 3, "expected HWC tensor");
        const int input_C = static_cast<int>(input.size(2));
        
        if (input_C == 1) {
            return _process<1>(input, noise_sigmas);
        } else if (input_C == 3) {
            return _process<3>(input, noise_sigmas);
        } else {
            TORCH_CHECK(false, "input channels must be 1 or 3, got ", input_C);
        }
    }
    
    int get_overlap_factor() const override { return overlap_factor_; }
    

};

} // namespace

std::shared_ptr<Wiener> create_wiener(torch::Device device,
    int width, int height, int overlap_factor, int tile_size) {

    if (tile_size == 16) {
        return std::make_shared<WienerImpl<16>>(device,  overlap_factor);
    } else {
        return std::make_shared<WienerImpl<32>>(device,  overlap_factor);
    }
}





