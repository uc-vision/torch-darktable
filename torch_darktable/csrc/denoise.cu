#include <cuda_runtime.h>

#include "reduction.h"
#include "denoise.h"
#include "fft.h"



namespace {

constexpr int K = 32;  // back to full block size
constexpr int C = 3;  // fixed number of channels for now

typedef Complex Tile[K*K];

__constant__ float d_window[K*K];      // analysis window
__constant__ float d_interp_window[K*K]; // synthesis window

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

__device__ __forceinline__ int get_tile_idx() {
    auto block = cg::this_thread_block();
    dim3 t = block.thread_index();
    return t.y * K + t.x;
}

__device__ __forceinline__ int reflect_index(int x, int limit) {
    if (x < 0) x = -x - 1;
    if (x >= limit) x = 2 * limit - x - 1;
    return x;
}

__device__ void load_tile(
    const float* __restrict__ img,
    Tile tile_data,
    int c, int stride, int H, int W, int C
) {
    int2 g = get_group_pos();
    int2 t = get_thread_pos();
    int tile_idx = get_tile_idx();
    
    // Load one pixel with reflect padding
    int2 src_pos = g * stride + t;
    int2 refl_pos = make_int2(reflect_index(src_pos.x, W), reflect_index(src_pos.y, H));
    float val = img[(refl_pos.y * W + refl_pos.x) * C + c];  // HWC layout
    tile_data[tile_idx] = Complex(val, 0.0f);
}

__device__ void store_tile(
    float* __restrict__ out,
    float* __restrict__ mask,
    const Tile tile_data,
    int c, int stride, int H_pad, int W_pad, int C, float mean_val
) {
    int2 g = get_group_pos();
    int2 t = get_thread_pos();
    int tile_idx = get_tile_idx();
    
    // Store one pixel back to output
    int2 out_pos = g * stride + t + K;  // Add padding offset
    if (out_pos.y < H_pad && out_pos.x < W_pad) {
        float reconstructed = (tile_data[tile_idx].x + mean_val * d_window[tile_idx]) * d_interp_window[tile_idx];
        atomicAdd(&out[(out_pos.y * W_pad + out_pos.x) * C + c], reconstructed);  // HWC layout
        if (c == 0) {  // Only add to mask once per spatial position
            atomicAdd(&mask[out_pos.y * W_pad + out_pos.x], d_window[tile_idx] * d_interp_window[tile_idx]);
        }
    }
}


__device__ float apply_zero_mean_window(
    Tile tile_data) {

    int tile_idx = get_tile_idx();
    float mean_val = block_reduce_mean(tile_data[tile_idx].x, K * K);

    tile_data[tile_idx] = (tile_data[tile_idx] - mean_val) * d_window[tile_idx];
    return mean_val;
}



__device__ void apply_wiener_gain(
    Tile tile_data, 
    float noise_power, float eps
) {
    int tile_idx = get_tile_idx();

    Complex data = tile_data[tile_idx];

    float power = data.magnitude_squared() + eps;
    float gain = fmaxf(power - noise_power, 0.0f) / power;
    
    tile_data[tile_idx] = gain * data;
}


// Main kernel: orchestrates the Wiener filtering pipeline
__global__ void wiener_tile_kernel(
    const float* __restrict__ img,    // (C, H, W)
    float* __restrict__ out,          // (C, H_pad, W_pad)  
    float* __restrict__ mask,         // (H_pad, W_pad)
    int H, int W, int H_pad, int W_pad,
    int stride, int grid_h, int grid_w,
    float noise_power, float eps
) {

    __shared__ Tile tile_data;

    // Process all channels for this spatial tile
    for (int c = 0; c < C; c++) {
        // 1. Load tile with reflect padding
        load_tile(img, tile_data, c, stride, H, W, C);
        cg::this_thread_block().sync();
        
        // 2. Compute mean and apply windowing
        float mean_val = apply_zero_mean_window(tile_data);
        cg::this_thread_block().sync();
        
        // 7. Store tile back to output (add mean back)
        store_tile(out, mask, tile_data, c, stride, H_pad, W_pad, C, mean_val);
    }  // End channel loop
}

static torch::Tensor make_gaussian_window(int size, float weight, torch::Device device) {
    float half = size / 2.0f;
    auto r = torch::arange(-half + 0.5f, half - 0.5f + 1, torch::dtype(torch::kFloat32).device(device));
    auto w = torch::exp(-(r * r) / (weight * half * half));
    return w.unsqueeze(0) * w.unsqueeze(1);
}

static void init_windows(float weight_fft, float weight_interp, torch::Device device) {
    auto window = make_gaussian_window(K, weight_fft, device);
    auto interp_window = make_gaussian_window(K, weight_interp, device);
    
    // Copy to 1D constant arrays
    auto err1 = cudaMemcpyToSymbol(d_window, window.data_ptr<float>(), K * K * sizeof(float));
    auto err2 = cudaMemcpyToSymbol(d_interp_window, interp_window.data_ptr<float>(), K * K * sizeof(float));
    
    if (err1 != cudaSuccess) {
        throw std::runtime_error("Failed to copy window to constant memory: " + std::string(cudaGetErrorString(err1)));
    }
    if (err2 != cudaSuccess) {
        throw std::runtime_error("Failed to copy interp_window to constant memory: " + std::string(cudaGetErrorString(err2)));
    }
}

__global__ void normalize_and_crop_kernel(
    const float* __restrict__ padded_out,  // (C, H_pad, W_pad)
    const float* __restrict__ mask,        // (H_pad, W_pad)
    float* __restrict__ final_out,         // (C, H, W) - final output
    int H, int W, int H_pad, int W_pad, int K, float eps
) {
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, 
                         blockIdx.y * blockDim.y + threadIdx.y);
    
    if (pos.y < H && pos.x < W) {
        int2 pad_pos = pos + K;
        int mask_idx = pad_pos.y * W_pad + pad_pos.x;
        float mask_val = mask[mask_idx] + eps;
        
        // Process all channels for this spatial position
        for (int c = 0; c < C; c++) {
            int padded_idx = (pad_pos.y * W_pad + pad_pos.x) * C + c;  // HWC layout
            int final_idx = (pos.y * W + pos.x) * C + c;  // HWC layout
            
            float out_val = padded_out[padded_idx] + eps;
            final_out[final_idx] = out_val / mask_val;
        }
    }
}


struct CudaWiener final : public Wiener {
    CudaWiener(torch::Device device, int width, int height, float sigma, float eps)
        : device_(device), width_(width), height_(height), sigma_(sigma), eps_(eps) {
        
        constexpr float weight_fft = 0.3f;
        constexpr float weight_interp = 0.3f;
        init_windows(weight_fft, weight_interp, device);
    }
    
    torch::Tensor process(const torch::Tensor &input) override {
        TORCH_CHECK(input.device() == device_, "input device mismatch");
        TORCH_CHECK(input.dim() == 3, "expected HWC tensor");

        const int H = static_cast<int>(input.size(0));
        const int W = static_cast<int>(input.size(1));
        const int C = static_cast<int>(input.size(2));
        TORCH_CHECK(H == height_ && W == width_, "input size mismatch");
        
        // Compute dimensions on-demand
        const int stride = K / 4;  // K/4 overlap = 8
        const int h_pad = H + 2 * K;
        const int w_pad = W + 2 * K;
        const int grid_h = (H + stride - 1) / stride;  // Cover input space
        const int grid_w = (W + stride - 1) / stride;
        
        auto padded_out = torch::zeros({h_pad, w_pad, C}, input.options());
        auto mask = torch::zeros({h_pad, w_pad}, input.options());
        auto final_out = torch::zeros({H, W, C}, input.options());
        
        // Noise power: compute actual sum of window squares like Python
        constexpr float weight_fft = 0.3f;
        auto window = make_gaussian_window(K, weight_fft, device_);
        float window_sum_sq = torch::sum(window * window).item<float>();
        float noise_power = window_sum_sq * (sigma_ * sigma_);
        
        dim3 grid(grid_w, grid_h);
        dim3 block(K, K);  // 32x32 threads, each handles one pixel

        
        wiener_tile_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            padded_out.data_ptr<float>(),
            mask.data_ptr<float>(),
            H, W, h_pad, w_pad,
            stride, grid_h, grid_w,
            noise_power, eps_
        );
        
        // Check for kernel launch errors
        auto cuda_err = cudaGetLastError();
        TORCH_CHECK(cuda_err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(cuda_err));
        
        // Normalize and crop in a single kernel
        dim3 norm_grid((W + 31) / 32, (H + 31) / 32);
        dim3 norm_block(32, 32);
        
        normalize_and_crop_kernel<<<norm_grid, norm_block>>>(
            padded_out.data_ptr<float>(),
            mask.data_ptr<float>(),
            final_out.data_ptr<float>(),
            H, W, h_pad, w_pad, K, eps_
        );
        
        // Synchronize to catch runtime errors
        cuda_err = cudaDeviceSynchronize();
        TORCH_CHECK(cuda_err == cudaSuccess, "Kernel execution failed: ", cudaGetErrorString(cuda_err));
        
        return final_out;
    }
    
    void set_sigma(float sigma) override { sigma_ = sigma; }
    float get_sigma() const override { return sigma_; }
    void set_eps(float eps) override { eps_ = eps; }
    float get_eps() const override { return eps_; }
    
private:
    torch::Device device_;
    int width_, height_;
    float sigma_, eps_;
};

} // namespace

std::shared_ptr<Wiener> create_wiener(torch::Device device,
    int width, int height, float sigma, float eps) {

    return std::make_shared<CudaWiener>(device, width, height, sigma, eps);
}




