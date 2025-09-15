/*
    This file is part of darktable,
    copyright (c) 2016 johannes hanika.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include "laplacian.h"
#include "../device_math.h"
#include "../cuda_utils.h"


// #define USE_CUDA_TIMER
#ifdef USE_CUDA_TIMER
    using Timer = CudaTimer;
#else
    using Timer = NullTimer;
#endif




// Device arrays for gamma pointer arrays
constexpr int max_gamma = 8;
__device__ half_t* d_gamma_level0[max_gamma];
__device__ half_t* d_gamma_level1[max_gamma];



// Helper function for downsampling
inline int dl(int x, int level) { return (x + (1 << level) - 1) >> level; }


__device__ int2 clamp_boundary(int2 pos, int2 size) {
    // Handle right/bottom boundaries based on odd/even dimensions
    if(size.x & 1) { if(pos.x > size.x-2) pos.x = size.x-2; }
    else           { if(pos.x > size.x-3) pos.x = size.x-3; }
    if(size.y & 1) { if(pos.y > size.y-2) pos.y = size.y-2; }
    else           { if(pos.y > size.y-3) pos.y = size.y-3; }
    
    // Handle left/top boundaries
    if(pos.x <= 0) pos.x = 1;
    if(pos.y <= 0) pos.y = 1;
    
    return pos;
}


__global__ void pad_input(
    float* input,
    float* padded,
    const int2 input_size,                  // dimensions of input
    const int max_supp,            // size of border
    const int2 padded_size)                 // padded dimensions
{
    const int2 pos = get_thread_pos();
    int2 c = pos - max_supp;

    if(pos.x >= padded_size.x || pos.y >= padded_size.y) return;
    // fill boundary with max_supp px:
    if(c.x >= input_size.x) c.x = input_size.x-1;
    if(c.y >= input_size.y) c.y = input_size.y-1;
    if(c.x < 0) c.x = 0;
    if(c.y < 0) c.y = 0;

    float pixel_val = read_imagef(input, c, input_size);
    write_imagef(padded, pos, padded_size, pixel_val);
}

// Pad input directly to fp16 output
__global__ void pad_input_half(
    float* input,
    half_t* padded,
    const int2 input_size,                  // dimensions of input
    const int max_supp,            // size of border
    const int2 padded_size)                 // padded dimensions
{
    const int2 pos = get_thread_pos();
    int2 c = pos - max_supp;

    if(pos.x >= padded_size.x || pos.y >= padded_size.y) return;
    // fill boundary with max_supp px:
    if(c.x >= input_size.x) c.x = input_size.x-1;
    if(c.y >= input_size.y) c.y = input_size.y-1;
    if(c.x < 0) c.x = 0;
    if(c.y < 0) c.y = 0;

    float_t pixel_val = read_imagef(input, c, input_size);
    write_imagef_half(padded, pos, padded_size, pixel_val);
}

__device__ float_t expand_gaussian(
    half_t* coarse,
    const int2 pos,
    const int2 fine_size,
    const int2 coarse_size)
{
    constexpr float_t w[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};
    const int2 coarse_pos = pos / 2;
    
    const int x_odd = pos.x & 1;
    const int y_odd = pos.y & 1;
    
    const int i_start = x_odd ? 0 : -1;
    const int i_end = x_odd ? 1 : 1;
    const int j_start = y_odd ? 0 : -1; 
    const int j_end = y_odd ? 1 : 1;
    
    float_t c = 0.0f;
    #pragma unroll
    for(int i = i_start; i <= i_end; i++) {
        #pragma unroll
        for(int j = j_start; j <= j_end; j++) {
            int2 access_pos = coarse_pos + make_int2(i, j);
            float_t pixel = read_imagef_half(coarse, access_pos, coarse_size);
            const int wi = x_odd ? (2*i+1) : (2*i+2);
            const int wj = y_odd ? (2*j+1) : (2*j+2);
            c += pixel * w[wi] * w[wj];
        }
    }
    return 4.0f * c;
}

// Gauss reduce from fp32 input to fp16 output
__global__ void
gauss_reduce(
    float* input,                   // fine input buffer (fp32)
    half_t* coarse,                 // coarse scale, blurred output buf (fp16)
    const int2 coarse_size,         // coarse res
    const int2 input_size)
{
    const int2 pos = get_thread_pos();
    int2 c = pos;

    if(pos.x >= coarse_size.x || pos.y >= coarse_size.y) return;
    // fill boundary with 1 px:
    if(pos.x >= coarse_size.x-1) c.x = coarse_size.x-2;
    if(pos.y >= coarse_size.y-1) c.y = coarse_size.y-2;
    if(c.x <= 0) c.x = 1;
    if(c.y <= 0) c.y = 1;

    // blur, store only coarse res
    float_t pixel_val = 0.0f;
    constexpr float_t w[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};
    // direct 5x5 stencil only on required pixels:
    #pragma unroll
    for(int j=-2;j<=2;j++) {
      #pragma unroll
      for(int i=-2;i<=2;i++) {
        pixel_val += read_imagef(input, 2*c + make_int2(i, j), input_size) * w[i+2] * w[j+2];
      }
    }

    write_imagef_half(coarse, pos, coarse_size, pixel_val);
}

// Gauss reduce from fp16 input to fp16 output (for pyramid levels > 1)
__global__ void
gauss_reduce_half(
    half_t* input,                  // fine input buffer (fp16)
    half_t* coarse,                 // coarse scale, blurred output buf (fp16)
    const int2 coarse_size,         // coarse res
    const int2 input_size)
{
    const int2 pos = get_thread_pos();
    int2 c = pos;

    if(pos.x >= coarse_size.x || pos.y >= coarse_size.y) return;
    // fill boundary with 1 px:
    if(pos.x >= coarse_size.x-1) c.x = coarse_size.x-2;
    if(pos.y >= coarse_size.y-1) c.y = coarse_size.y-2;
    if(c.x <= 0) c.x = 1;
    if(c.y <= 0) c.y = 1;

    // blur, store only coarse res
    float_t pixel_val = 0.0f;
    constexpr float_t w[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};
    // direct 5x5 stencil only on required pixels:
    #pragma unroll
    for(int j=-2;j<=2;j++) {
      #pragma unroll
      for(int i=-2;i<=2;i++) {
        pixel_val += read_imagef_half(input, 2*c + make_int2(i, j), input_size) * w[i+2] * w[j+2];
      }
    }

    write_imagef_half(coarse, pos, coarse_size, pixel_val);
}

__device__ inline float_t laplacian(
    half_t* tex_coarse,  // coarse res gaussian (fp16)
    half_t* tex_fine,    // fine res gaussian (fp16)
    const int2 pos,                     // fine index
    const int2 clamped_pos,                    // clamped fine index
    const int2 fine_size,                    // fine size
    const int2 coarse_size)             // coarse size
{
    const float_t c = expand_gaussian(tex_coarse, clamped_pos, fine_size, coarse_size);
    return read_imagef_half(tex_fine, pos, fine_size) - c;
}

template<int num_gamma>
__global__ void laplacian_assemble(
    half_t* tex_input,    // original input buffer, gauss at current fine pyramid level
    half_t* tex_output1,  // state of reconstruction, coarse output buffer
    half_t* output0,      // reconstruction, one level finer, run kernel on this dimension
    const int2 fine_size)       // width and height of the fine buffers (l0)
{    
    const int2 pos = get_thread_pos();
    const int w = fine_size.x, h = fine_size.y;

    if(pos.x >= w || pos.y >= h) return;
    
    const int2 clamped_pos = clamp_boundary(pos, fine_size);
    const int2 coarse_size = make_int2((fine_size.x-1)/2+1, (fine_size.y-1)/2+1);
    
    float_t pixel_val = expand_gaussian(tex_output1, clamped_pos, fine_size, coarse_size);

    const float_t v = read_imagef_half(tex_input, pos, fine_size);
    int hi = 1;
    // what we mean is this:
    // for(;hi<num_gamma-1 && gamma[hi] <= v;hi++);
    for(;hi<num_gamma-1 && ((float)hi+.5f)/(float)num_gamma <= v;hi++);
    int lo = hi-1;
    // const float a = fminf(fmaxf((v - gamma[lo])/(gamma[hi]-gamma[lo]), 0.0f), 1.0f);
    const float a = fminf(fmaxf(v*num_gamma - ((float)lo+.5f), 0.0f), 1.0f);
    
    float_t l0 = laplacian(d_gamma_level1[lo], d_gamma_level0[lo], pos, clamped_pos, fine_size, coarse_size);
    float_t l1 = laplacian(d_gamma_level1[lo+1], d_gamma_level0[lo+1], pos, clamped_pos, fine_size, coarse_size);

    pixel_val += l0 * (1.0f-a) + l1 * a;
    write_imagef_half(output0, pos, fine_size, pixel_val);
}

// Fast exp approximation for range [-8, 0] (good for Gaussian-like terms)
__device__ inline float fast_expf(float x) {
    // Clamp to safe range
    x = fmaxf(x, -8.0f);
    
    // Use polynomial approximation: exp(x) ≈ (1 + x/8)^8 for x in [-8,0]
    const float a = 1.0f + x * 0.125f; // x/8
    const float a2 = a * a;
    const float a4 = a2 * a2;
    return a4 * a4; // a^8
}

__device__ float curve(
    const float x,
    const float g,
    const float sigma,
    const float shadows,
    const float highlights,
    const float clarity)
{

    const float c = x-g;
    float val;
    const float ssigma = c > 0.0f ? sigma : - sigma;
    const float shadhi = c > 0.0f ? shadows : highlights;
    if (fabsf(c) > 2*sigma) val = g + ssigma + shadhi * (c-ssigma); // linear part
    else
    { // blend in via quadratic bezier
        const float t = clipf(c / (2.0f*ssigma));
        const float t2 = t * t;
        const float mt = 1.0f-t;
        val = g + ssigma * 2.0f*mt*t + t2*(ssigma + ssigma*shadhi);
    }
    const float exp_arg = -c*c/(2.0f*sigma*sigma/3.0f);
    val += clarity * c * expf(exp_arg);
    return val;
}

// Process curve to fp32 output (for level 0)
__global__ void
process_curve(
    float* tex_input,
    float* output,
    const float g,
    const float sigma,
    const float shadows,
    const float highlights,
    const float clarity,
    const int2 size)
{
    const int2 pos = get_thread_pos();
    if(pos.x >= size.x || pos.y >= size.y) return;

    float_t pixel_val = read_imagef(tex_input, pos, size);
    pixel_val = curve(pixel_val, g, sigma, shadows, highlights, clarity);
    write_imagef(output, pos, size, pixel_val);
}

// Process curve from fp16 input to fp16 output
__global__ void
process_curve_half(
    half_t* tex_input,
    half_t* output,
    const float g,
    const float sigma,
    const float shadows,
    const float highlights,
    const float clarity,
    const int2 size)
{
    const int2 pos = get_thread_pos();
    if(pos.x >= size.x || pos.y >= size.y) return;

    float_t pixel_val = read_imagef_half(tex_input, pos, size);
    pixel_val = curve(pixel_val, g, sigma, shadows, highlights, clarity);
    write_imagef_half(output, pos, size, pixel_val);
}

// Multi-gamma curve processing to fp16
template<int num_gamma>
__global__ void
process_curves_batch(
    float* tex_input,
    half_t** outputs,            // Array of fp16 output pointers 
    const float sigma,
    const float shadows,
    const float highlights,
    const float clarity,
    const int2 size)
{
    const int2 pos = get_thread_pos();
    if(pos.x >= size.x || pos.y >= size.y) return;

    float_t input_val = read_imagef(tex_input, pos, size);
    
    // Process all gamma levels at once (compute in fp32, store as fp16)
    #pragma unroll
    for(int k = 0; k < num_gamma; k++) {
        const float_t g = (k + 0.5f) / (float_t)num_gamma;
        float_t curved_val = curve(input_val, g, sigma, shadows, highlights, clarity);
        write_imagef_half(outputs[k], pos, size, curved_val);
    }
}

__global__ void write_back(
    float* tex_processed,
    float* output,
    const int max_supp,
    const int2 output_size,
    const int2 processed_size)
{
    const int2 pos = get_thread_pos();
    if(pos.x >= output_size.x || pos.y >= output_size.y) return;

    float processed_val = read_imagef(tex_processed, pos + max_supp, processed_size);
    write_imagef(output, pos, output_size, processed_val);
}

__global__ void write_back_half(
    half_t* tex_processed,
    float* output,
    const int max_supp,
    const int2 output_size,
    const int2 processed_size)
{
    const int2 pos = get_thread_pos();
    if(pos.x >= output_size.x || pos.y >= output_size.y) return;

    float_t processed_val = read_imagef_half(tex_processed, pos + max_supp, processed_size);
    write_imagef(output, pos, output_size, processed_val);
}






// Complete LaplacianImpl class definition with all methods inline
template<int num_gamma = 6>
struct LaplacianImpl : public Laplacian
{
    torch::Device device_;
    int width, height;
    int num_levels;
    float sigma, highlights, shadows, clarity;
    int max_supp;
    int bwidth, bheight;

    std::vector<torch::Tensor> dev_padded;
    std::vector<torch::Tensor> dev_output;
    std::vector<std::vector<torch::Tensor>> dev_processed;
    

    // Constructor
    LaplacianImpl(torch::Device device, const int width, const int height, const float sigma,
                      const float shadows, const float highlights, const float clarity)
        : device_(device), width(width), height(height), sigma(sigma), shadows(shadows),
          highlights(highlights), clarity(clarity)
    {
        assert(width > 0 && height > 0);

        num_levels = std::min(max_levels, 31 - __builtin_clz(std::min(width, height)));
        max_supp = 1 << (num_levels - 1);
        bwidth = width + 2 * max_supp;
        bheight = height + 2 * max_supp;

        dev_processed.resize(num_gamma);
        
        // Defer buffer allocation until first process() call
        dev_padded.clear();
        dev_output.clear();
        for(int k = 0; k < num_gamma; k++) dev_processed[k].clear();
    }

    // Helper functions - all levels are fp16
    half_t* processed_ptr(int level, int k) { 
          return dev_processed[k][level].data_ptr<half_t>(); 
    }
    
    half_t* padded_ptr(int level) { 
        return dev_padded[level].data_ptr<half_t>(); 
    }
    
    half_t* output_ptr(int level) { 
      return dev_output[level].data_ptr<half_t>();  
    }

    // Processing function
    torch::Tensor process(const torch::Tensor& input) override
    {
        // Validate input tensor
        TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
        TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
        TORCH_CHECK(input.size(0) == height && input.size(1) == width, 
                    "Input tensor dimensions must match workspace dimensions");
        TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");

        
        // Ensure buffers are allocated before first use
        if (dev_padded.empty()) {
            allocate_buffers();
        }

        auto stream = c10::cuda::getCurrentCUDAStream();

        // Create output tensor
        auto output = torch::empty({height, width}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        
        // Execute processing steps with timing
        
        Timer timer(stream.stream());
        timer.record("pad_input_step");
        pad_input_step(input);
        timer.record("build_gaussian_pyramid_step");
        build_gaussian_pyramid_step();
        timer.record("process_gamma_curves_step");
        process_gamma_curves_step();
        timer.record("assemble_pyramid_step");
        assemble_pyramid_step();
        timer.record("write_back_step");
        write_back_step(output);
        timer.print_timings();
    
        
        return output;
    }
    
    // Buffer allocation
    void allocate_buffers()
    {
        for(int l = 0; l < num_levels; l++)
        {
            const int level_width = dl(bwidth, l);
            const int level_height = dl(bheight, l);

            dev_padded.push_back(torch::empty({level_height, level_width}, torch::device(torch::kCUDA).dtype(torch::kFloat16)));
            dev_output.push_back(torch::empty({level_height, level_width}, torch::device(torch::kCUDA).dtype(torch::kFloat16)));

            if (l == 0) {
                for(int k = 0; k < num_gamma; k++) dev_processed[k].clear();
            }
            for(int k = 0; k < num_gamma; k++) {
                auto tensor = torch::empty({level_height, level_width}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
                dev_processed[k].push_back(tensor);
            }
        }
    }
    
    // Helper functions for processing steps
    void pad_input_step(const torch::Tensor& input)
    {
        auto stream = c10::cuda::getCurrentCUDAStream();

        
        // Pad directly to fp16 storage
        pad_input_half<<<grid2d(bwidth, bheight), block_size_2d, 0, stream.stream()>>>(
            input.data_ptr<float>(), padded_ptr(0),
            make_int2(width, height), max_supp,
            make_int2(bwidth, bheight));
    }

    void build_gaussian_pyramid_step()
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        
        for(int l=1; l<num_levels; l++) {
            const int level_width = dl(bwidth, l);
            const int level_height = dl(bheight, l);
            
            // All levels now use fp16 -> fp16 processing
            gauss_reduce_half<<<grid2d(level_width, level_height), block_size_2d, 0, stream.stream()>>>(
                padded_ptr(l-1), 
                (l == num_levels-1) ? output_ptr(l) : padded_ptr(l), 
                make_int2(level_width, level_height), make_int2(dl(bwidth, l-1), dl(bheight, l-1)));
        }
    }

    void process_gamma_curves_step()
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        
        for(int k=0; k<num_gamma; k++) {
            const float g = (k + 0.5f) / (float)num_gamma;
            
            // Process curve directly: fp16 input -> fp16 output
            process_curve_half<<<grid2d(bwidth, bheight), block_size_2d, 0, stream.stream()>>>(
                padded_ptr(0), processed_ptr(0, k), 
                g, sigma, shadows, highlights, clarity,
                make_int2(bwidth, bheight));

            // Create gaussian pyramids for this gamma level
            for(int l=1; l<num_levels; l++) {
                const int level_width = dl(bwidth, l);
                const int level_height = dl(bheight, l);
                
                gauss_reduce_half<<<grid2d(level_width, level_height), block_size_2d, 0, stream.stream()>>>(
                    processed_ptr(l-1, k), processed_ptr(l, k), 
                    make_int2(level_width, level_height), make_int2(dl(bwidth, l-1), dl(bheight, l-1)));
            }
        }
    }

    void assemble_pyramid_step()
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        
        for(int l=num_levels-2; l >= 0; l--) {
            const int pw = dl(bwidth, l);
            const int ph = dl(bheight, l);
            
            // Fill pointer arrays on host
            std::vector<int64_t> h_ptrs0(num_gamma), h_ptrs1(num_gamma);
            for(int k = 0; k < num_gamma; k++) {
                h_ptrs0[k] = reinterpret_cast<int64_t>(processed_ptr(l, k));
                h_ptrs1[k] = reinterpret_cast<int64_t>(processed_ptr(l+1, k));
            }
            
            // Copy directly to device symbols  
            CUDA_CHECK(cudaMemcpyToSymbol(d_gamma_level0, h_ptrs0.data(), num_gamma * sizeof(half_t*)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_gamma_level1, h_ptrs1.data(), num_gamma * sizeof(half_t*)));
            
            laplacian_assemble<num_gamma><<<grid2d(pw, ph), block_size_2d, 0, stream.stream()>>>(
                padded_ptr(l), 
                output_ptr(l+1),
                output_ptr(l),
                make_int2(pw, ph)); 
        }
    }

    void write_back_step(torch::Tensor& output)
    {
        auto stream = c10::cuda::getCurrentCUDAStream();

        write_back_half<<<grid2d(width, height), block_size_2d, 0, stream.stream()>>>(
            output_ptr(0), output.data_ptr<float>(),
            max_supp, make_int2(width, height), make_int2(bwidth, bheight));
    }
    
    // Parameter inspection
    
    
    // Read accessors for properties
    float get_sigma() const override { return sigma; }
    float get_shadows() const override { return shadows; }
    float get_highlights() const override { return highlights; }
    float get_clarity() const override { return clarity; }

    // Adjustable parameter setters
    void set_sigma(float new_sigma) override { sigma = new_sigma; }
    void set_shadows(float new_shadows) override { shadows = new_shadows; }
    void set_highlights(float new_highlights) override { highlights = new_highlights; }
    void set_clarity(float new_clarity) override { clarity = new_clarity; }
};


// Explicit template instantiation for common values
template struct LaplacianImpl<4>;
template struct LaplacianImpl<6>;
template struct LaplacianImpl<8>;

// Factory function for workspace creation
std::shared_ptr<Laplacian> create_laplacian(
    torch::Device device, 
    int width, int height, 
    int num_gamma, 
    float sigma, float shadows, float highlights, float clarity) {

    switch(num_gamma) {
        // case 4:
        //     return std::make_shared<LaplacianImpl<4>>(device, width, height, sigma, shadows, highlights, clarity);
        case 6:
            return std::make_shared<LaplacianImpl<6>>(device, width, height, sigma, shadows, highlights, clarity);
        // case 8:
        //     return std::make_shared<LaplacianImpl<8>>(device, width, height, sigma, shadows, highlights, clarity);
        default:
            throw std::runtime_error("Unsupported gamma count: " + std::to_string(num_gamma));
    }
}

