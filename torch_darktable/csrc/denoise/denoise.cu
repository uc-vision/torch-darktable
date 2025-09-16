#include <cuda_runtime.h>

#include "reduction.h"
#include "denoise.h"
#include "fft.h"
#include "window.h"
#include "cuda_utils.h"


namespace {

constexpr float eps = 1e-15f;

template<int C>
struct pixel_type;

template<>
struct pixel_type<1> {
  using type = float;

  __device__ __forceinline__ static void atomic_add(float* addr, float value) {
      atomicAdd(addr, value);
  }

  __device__ __forceinline__ static float zero() {
    return 0.0f;
  }

  __host__ static float from_tensor(const torch::Tensor& tensor) {
    return tensor[0].item<float>();
  }

  __device__ __forceinline__ static float get(const float& pixel, int i) {
      return pixel;
  }

  __device__ __forceinline__ static void set(float& pixel, int i, float value) {
      pixel = value;
  }
};

template<>
struct pixel_type<3> {
  using type = float3;

  __device__ __forceinline__ static void atomic_add(float3* addr, float3 value) {
      atomicAdd(&addr->x, value.x);
      atomicAdd(&addr->y, value.y);
      atomicAdd(&addr->z, value.z);
  }

  __device__ __forceinline__ static float3 zero() {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  __host__ static float3 from_tensor(const torch::Tensor& tensor) {
      return make_float3(tensor[0].item<float>(), tensor[1].item<float>(), tensor[2].item<float>());
  }

  __device__ __forceinline__ static float get(const float3& pixel, int i) {
      return (&pixel.x)[i];
  }

  __device__ __forceinline__ static void set(float3& pixel, int i, float value) {
      (&pixel.x)[i] = value;
  }
};

template<int C>
using pixel_t = typename pixel_type<C>::type;

template<int C>
__device__ __forceinline__ pixel_t<C> block_mean(pixel_t<C> value) {
    auto block = cg::this_thread_block();
    __shared__ typename pixel_type<C>::type sum;
    
    if (block.thread_rank() == 0) {
        sum = pixel_type<C>::zero();
    }
    
    auto group = cg::coalesced_threads();
    pixel_t<C> result = cg::reduce(group, value, cg::plus<pixel_t<C>>{});
    if (group.thread_rank() == 0) {
        pixel_type<C>::atomic_add(&sum, result);
    }
    block.sync();
    
    return sum / block.size();
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
__device__ __forceinline__ pixel_t<C> load_pixel(
    const pixel_t<C>* __restrict__ img,
    int stride, int H, int W
) {
    int2 g = get_group_pos();
    int2 t = get_thread_pos();
    int tile_idx = get_tile_idx<K>();

       
    // Load one pixel with reflect padding
    // Shift grid to start earlier for proper boundary coverage
    int2 grid_offset = {K / stride, K / stride};  // Shift by one tile
    int2 src_pos = (g - grid_offset) * stride + t;
    int2 refl_pos = make_int2(reflect_index(src_pos.x, W), reflect_index(src_pos.y, H));
    return img[(refl_pos.y * W + refl_pos.x)];  // HWC layout
}

template<int K, int C>
__device__ __forceinline__ void store_pixel(
    pixel_t<C>* __restrict__ out,
    float* __restrict__ mask,

    int stride, int H_pad, int W_pad, 
    pixel_t<C> const& value,
    pixel_t<C> const& mean,
    float fft_window
) {
    int2 g = get_group_pos();
    int2 t = get_thread_pos();
    int tile_idx = get_tile_idx<K>();
    
    auto interp_window = Window<K>::interp_window(t);
    
    // Store one pixel back to output  
    int2 grid_offset = {K / stride, K / stride};
    int2 out_pos = (g - grid_offset) * stride + t + K;
    if (out_pos.y < H_pad && out_pos.x < W_pad) {
        int out_idx = out_pos.y * W_pad + out_pos.x;

        pixel_t<C> reconstructed = (value + mean * fft_window) * interp_window;

        pixel_type<C>::atomic_add(&out[out_idx], reconstructed);
        atomicAdd(&mask[out_idx], fft_window * interp_window);
    }
}


__device__ __forceinline__ Complex apply_gain(Complex value, float noise_power) {
  float power = value.magnitude_squared() + eps;
  float gain = fmaxf(power - noise_power, 0.0f) / power;
  return gain * value;
}

// Main kernel: orchestrates the Wiener filtering pipeline
template<int K, int C>
__global__ void wiener_tile_kernel(
    const pixel_t<C>* __restrict__ img,    // (C, H, W)
    pixel_t<C>* __restrict__ out,          // (C, H_pad, W_pad)  
    float* __restrict__ mask,         // (H_pad, W_pad)
    int H, int W, int H_pad, int W_pad,
    int stride, int grid_h, int grid_w,
    const pixel_t<C> noise_sigmas  // (C,)
) {

    int2 t = get_thread_pos();

    // 1. Load tile and compute mean
    pixel_t<C> value = load_pixel<K, C>(img, stride, H, W);
    pixel_t<C> mean = block_mean<C>(value);
  
    float fft_window = Window<K>::fft_window(t);
    value = (value - mean) * fft_window;

    #pragma unroll
    for (int i = 0; i < C; i++) {
      auto complex = FFT<K>::fft_2d(Complex(pixel_type<C>::get(value, i), 0.0f));
      complex = apply_gain(complex, pixel_type<C>::get(noise_sigmas, i));
      pixel_type<C>::set(value, i, FFT<K>::ifft_2d(complex).re);
    }
    
    // 6. Store tile back to output (add mean back)
      store_pixel<K, C>(out, mask, stride, H_pad, W_pad, value, mean, fft_window);

}


template<int K, int C>
__global__ void normalize_and_crop_kernel(
    pixel_t<C> const * __restrict__ padded_out,  // (C, H_pad, W_pad)
    pixel_t<C>* __restrict__ final_out,         // (C, H, W) - final output
    float const* __restrict__ mask,        // (H_pad, W_pad)

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


template<int K, int C = 3>
struct WienerImpl final : public Wiener {
    static_assert(K == 16 || K == 32, "K must be 16 or 32");
    static_assert(C == 1 || C == 3, "C must be 1 or 3");
    
    typedef typename pixel_type<C>::type pixel;

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
        dim3 block(K, K);  // KxK threads, each handles one pixel

        pixel const noise_sigmas_val = pixel_type<C>::from_tensor(noise_sigmas);
       
        wiener_tile_kernel<K, C><<<grid, block>>>(
            reinterpret_cast<pixel const*>(input.data_ptr<float>()),
            reinterpret_cast<pixel*>(padded_out.data_ptr<float>()),
            mask.data_ptr<float>(),
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
            reinterpret_cast<pixel const*>(padded_out.data_ptr<float>()),
            reinterpret_cast<pixel*>(final_out.data_ptr<float>()),
            mask.data_ptr<float>(),
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





