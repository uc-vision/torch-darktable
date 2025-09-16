
#include <cooperative_groups.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/complex.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>

namespace cg = cooperative_groups;

#include "reduction.h"
#include "denoise.h"

#include "fft.h"
#include "window.h"


namespace {

constexpr float eps = 1e-15f;

// Simple utility function
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// Clean type aliases - half precision for storage, float for accumulators
// using half_t = __half;
using half_t = at::Half;
using Complex = cutlass::complex<half_t>;
template<typename T, int N>
using Array = cutlass::Array<T, N>;
template<int N>
using Half = cutlass::Array<half_t, N>;
template<int N>
using Float = cutlass::Array<float, N>;

template<int C>
using float_to_half = cutlass::NumericArrayConverter<float, half_t, C>;

template<int C>
using half_to_float = cutlass::NumericArrayConverter<half_t, float, C>;

// Generic block mean for any channel count - use float accumulators for numerical stability
template<int C>
__device__ __forceinline__ Half<C> block_mean(Half<C> value) {
    auto block = cg::this_thread_block();
    __shared__ Float<C> sum;
    
    if (block.thread_rank() == 0) {
        sum.clear();
    }

    auto group = cg::coalesced_threads();
    Float<C> result = cg::reduce(group, float_to_half<C>::convert(value), cg::plus<Float<C>>{});
    if (group.thread_rank() == 0) {
        for(int i = 0; i < C; i++) {
            atomicAdd(&sum[i], result[i]);
        }
    }
    block.sync();
    
    // Convert back to half precision using array operators
    return half_to_float<C>::convert(sum * (1.0f / block.size()));
}


// Using custom FFT implementation

__device__ __forceinline__ int2 get_group_pos() {
    auto block = cg::this_thread_block();
    dim3 g = block.group_index();
    return make_int2(g.x, g.y);
}

__device__ __forceinline__ int2 get_thread_pos() {
    auto block = cg::this_thread_block();
    dim3 t = block.thread_index();
    return make_int2(t.x, t.y);
}

__device__ __forceinline__ int reflect_index(int x, int limit) {
    if (x < 0) x = -x;
    if (x >= limit) x = 2 * limit - x - 1;
    return x;
}


// Device helper functions for the kernel
template<int K>
__device__ __forceinline__ int get_tile_idx() {
    auto block = cg::this_thread_block();
    dim3 t = block.thread_index();
    return t.y * K + t.x;
}

template<int K, int C>
__device__ __forceinline__ Half<C> load_pixel(
    const Half<C>* __restrict__ img,
    int stride, int H, int W
) {
    int2 g = get_group_pos();
    int2 t = get_thread_pos();
       
    // Load one pixel with reflect padding
    // Shift grid to start earlier for proper boundary coverage
    int2 grid_offset = {K / stride, K / stride};  // Shift by one tile
    int2 src_pos = (g - grid_offset) * stride + t;
    int2 refl_pos = make_int2(reflect_index(src_pos.x, W), reflect_index(src_pos.y, H));
    return img[(refl_pos.y * W + refl_pos.x)];  // HWC layout
}



// Generic store operation for any channel count  
template<int K, int C>
__device__ __forceinline__ void store_pixel(
    Half<C>* __restrict__ out,
    half_t* __restrict__ mask,
    int stride, int H_pad, int W_pad, 
    Half<C> const& value,
    Half<C> const& mean,
    half_t fft_window
) {


    int2 g = get_group_pos();
    int2 t = get_thread_pos();
    
    auto interp_window = Window<K>::interp_window(t);
    
    // Store one pixel back to output  
    int2 grid_offset = {K / stride, K / stride};
    int2 out_pos = (g - grid_offset) * stride + t + K;
    if (out_pos.y < H_pad && out_pos.x < W_pad) {
        int out_idx = out_pos.y * W_pad + out_pos.x;

        Half<C> reconstructed = (value + mean * fft_window) * interp_window;

        for(int i = 0; i < C; i++) {
            atomicAdd(&out[out_idx][i], reconstructed[i]);
        }

        atomicAdd(&mask[out_idx], fft_window * interp_window);
    }
}


__device__ __forceinline__ Complex apply_gain(Complex value, half_t sigma) {
  auto power = cutlass::norm(value) + eps;
  auto gain = fmaxf(power - sigma * sigma, 0.0f) / power;
  
  return value * gain;
}

// Main kernel: orchestrates the Wiener filtering pipeline
template<int K, int C>
__global__ void wiener_tile_kernel(
    const Half<C>* __restrict__ img,    // (C, H, W)
    Half<C>* __restrict__ out,          // (C, H_pad, W_pad) - half precision  
    half_t* __restrict__ mask,         // (H_pad, W_pad)
    int H, int W, int H_pad, int W_pad,
    int stride, int grid_h, int grid_w,
    const Half<C> noise_sigmas  // (C,)
) {

    int2 t = get_thread_pos();

    // 1. Load tile and compute mean
    Half<C> value = load_pixel<K, C>(img, stride, H, W);
    Half<C> mean = block_mean<C>(value);
  
    half_t fft_window = Window<K>::fft_window(t);
    Half<C> windowed_value = (value - mean) * fft_window;

    #pragma unroll
    for (int i = 0; i < C; i++) {
      auto complex = FFT<K>::fft_2d(Complex(windowed_value[i], 0.0_hf));
      complex = apply_gain(complex, noise_sigmas[i]);
      windowed_value[i] = FFT<K>::ifft_2d(complex).real();
    }
    
    // 6. Store tile back to output (add mean back)
    store_pixel<K, C>(out, mask, stride, H_pad, W_pad, windowed_value, mean, fft_window);

}


template<int K, int C>
__global__ void normalize_and_crop_kernel(
    Half<C> const * __restrict__ padded_out,  // (C, H_pad, W_pad) - half precision
    Half<C>* __restrict__ final_out,         // (C, H, W) - final output in half
    half_t const* __restrict__ mask,        // (H_pad, W_pad)

    int H, int W, int H_pad, int W_pad
) {
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, 
                         blockIdx.y * blockDim.y + threadIdx.y);
    
    if (pos.y < H && pos.x < W) {
        int2 pad_pos = pos + K;
        int padded_idx = pad_pos.y * W_pad + pad_pos.x;
        int idx = (pos.y * W + pos.x);  // HWC layout

        // Normalize half precision values using float arithmetic
        Half<C> pixel = padded_out[padded_idx];
        auto divisor = 1.0f / (mask[padded_idx] + eps);
        Half<C> result = pixel * divisor;
        final_out[idx] = result;
    }
}


// Helper functions for tensor conversion
template<int C>
__host__ Half<C> tensor_to_array(const torch::Tensor& tensor) {
    Half<C> result;
    for (int i = 0; i < C; i++) {
        result[i] = tensor[i].item<at::Half>();
    }
    return result;
}

template<int K, int C = 3>
struct WienerImpl final : public Wiener {
    static_assert(K == 16 || K == 32, "K must be 16 or 32");
    static_assert(C == 1 || C == 3, "C must be 1 or 3");
    
    using Pixel = Half<C>;

    torch::Device device_;
    int overlap_factor_;

public:
    WienerImpl(torch::Device device,
      int overlap_factor = 4,  
      const float interp_scale = 0.3f, const float fft_scale = 0.3f)
        : device_(device), overlap_factor_(overlap_factor) {
        
        Window<K>::init(fft_scale, interp_scale);
    }
    
    
    torch::Tensor process(const torch::Tensor &input, const torch::Tensor &noise_sigmas) override {
        TORCH_CHECK(input.device() == device_, "input device mismatch");
        TORCH_CHECK(input.dim() == 3, "expected HWC tensor");
        TORCH_CHECK(input.dtype() == torch::kHalf, "input must be half precision (float16)");

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
        
        // All tensors in half precision for memory efficiency
        auto padded_out = torch::zeros({h_pad, w_pad, C}, input.options());
        auto mask = torch::zeros({h_pad, w_pad}, input.options());
        auto final_out = torch::empty({H, W, C}, input.options());
        
        // Process noise sigmas
        TORCH_CHECK(noise_sigmas.numel() == C, "noise_sigmas must have C elements");
        
        dim3 grid(grid_w, grid_h);
        dim3 block(K, K);  // KxK threads, each handles one pixel

        Pixel noise_sigmas_val = tensor_to_array<C>(noise_sigmas);
       
        wiener_tile_kernel<K, C><<<grid, block>>>(
            reinterpret_cast<Pixel const*>(input.data_ptr<torch::Half>()),
            reinterpret_cast<Pixel*>(padded_out.data_ptr<torch::Half>()),
            mask.data_ptr<torch::Half>(),
            H, W, h_pad, w_pad,
            stride, grid_h, grid_w,
            noise_sigmas_val
        );

        
        // Check for kernel launch errors
        auto cuda_err = cudaGetLastError();
        TORCH_CHECK(cuda_err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(cuda_err));
        
        // Normalize and crop in a single kernel
        dim3 norm_grid(div_up(W, K), div_up(H, K));
        dim3 norm_block(K, K);
        
        normalize_and_crop_kernel<K, C><<<norm_grid, norm_block>>>(
            reinterpret_cast<Half<C> const*>(padded_out.data_ptr<torch::Half>()),
            reinterpret_cast<Half<C>*>(final_out.data_ptr<torch::Half>()),
            mask.data_ptr<torch::Half>(),
            H, W, h_pad, w_pad
        );
        
        // Synchronize to catch runtime errors
        cuda_err = cudaDeviceSynchronize();
        TORCH_CHECK(cuda_err == cudaSuccess, "Kernel execution failed: ", cudaGetErrorString(cuda_err));
        
        return final_out;
    }
    
    int get_overlap_factor() const override { return overlap_factor_; }
    

};

} // namespace

std::shared_ptr<Wiener> create_wiener(torch::Device device,
    int width, int height, int overlap_factor, int tile_size, int channels) {

    if (tile_size == 16 && channels == 1) {
        return std::make_shared<WienerImpl<16, 1>>(device,  overlap_factor);
    } else if (tile_size == 16 && channels == 3) {
        return std::make_shared<WienerImpl<16, 3>>(device,  overlap_factor);
    } else if (tile_size == 32 && channels == 1) {
        return std::make_shared<WienerImpl<32, 1>>(device,  overlap_factor);
    } else {
        return std::make_shared<WienerImpl<32, 3>>(device,  overlap_factor);
    }
}





