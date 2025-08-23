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
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include "laplacian.h"
#include "device_math.h"

// Device arrays for gamma pointer arrays

constexpr int max_gamma = 8;
__device__ float* d_gamma_level0[max_gamma];
__device__ float* d_gamma_level1[max_gamma];

// CUDA error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

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

// OpenCL-style wrapper for texture reading (single channel)
__device__ float read_imagef(float* data, int2 coord, int2 size) {
    return data[coord.y * size.x + coord.x];
}

// Clamped version for boundary handling
__device__ float clamped_read_imagef(float* data, int2 coord, int2 size) {
    int x = coord.x;
    int y = coord.y;
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    if(x >= size.x) x = size.x - 1;
    if(y >= size.y) y = size.y - 1;
    
    int index = y * size.x + x;
    return data[index];
}

// OpenCL-style wrapper for image writing (single channel)
__device__ void write_imagef(float* output, int2 coord, int2 size, float value) {
    output[coord.y * size.x + coord.x] = value;
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

    float pixel_val = read_imagef(input, c, input_size) * 0.01f;
    write_imagef(padded, pos, padded_size, pixel_val);
}

__device__ float expand_gaussian(
    float* coarse,
    const int2 pos,
    const int2 fine_size,
    const int2 coarse_size)
{
    float c = 0.0f;
    const float w[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};
    const int2 coarse_pos = pos / 2;
    
    switch((pos.x&1) + 2*(pos.y&1))
    {
        case 0: // both are even, 3x3 stencil
            for(int i=-1;i<=1;i++) 
              for(int j=-1;j<=1;j++) {
                float pixel = clamped_read_imagef(coarse, coarse_pos + make_int2(i, j), coarse_size);
                c += pixel*w[2*j+2]*w[2*i+2];
            }
            break;
        case 1: // i is odd, 2x3 stencil
            for(int i=0;i<=1;i++) 
              for(int j=-1;j<=1;j++) {
                float pixel = clamped_read_imagef(coarse, coarse_pos + make_int2(i, j), coarse_size);
                c += pixel*w[2*j+2]*w[2*i+1];
            }
            break;
        case 2: // j is odd, 3x2 stencil
            for(int i=-1;i<=1;i++) 
              for(int j=0;j<=1;j++) {
                float pixel = clamped_read_imagef(coarse, coarse_pos + make_int2(i, j), coarse_size);
                c += pixel*w[2*j+1]*w[2*i+2];
            }
            break;
        default: // case 3: // both are odd, 2x2 stencil
            for(int i=0;i<=1;i++) 
              for(int j=0;j<=1;j++) {
                float pixel = clamped_read_imagef(coarse, coarse_pos + make_int2(i, j), coarse_size);
                c += pixel*w[2*j+1]*w[2*i+1];
            }
            break;
    }
    return 4.0f * c;
}

__global__ void
gauss_expand(
    float* coarse,                   // coarse input
    float* fine,                     // upsampled blurry output
    const int2 fine_size,                    // resolution of fine, also run kernel on fine res
    const int2 coarse_size)
{
    const int2 pos = get_thread_pos();
    int2 c = pos;

    if(pos.x >= fine_size.x || pos.y >= fine_size.y) return;
    
    c = clamp_boundary(pos, fine_size);

    float pixel_val = expand_gaussian(coarse, c, fine_size, coarse_size);
    write_imagef(fine, pos, fine_size, pixel_val);
}

__global__ void
gauss_reduce(
    float* input,                   // fine input buffer
    float* coarse,                  // coarse scale, blurred input buf
    const int2 coarse_size,                   // coarse res (also run this kernel on coarse res only)
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
    float pixel_val = 0.0f;
    const float w[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};
    // direct 5x5 stencil only on required pixels:
    for(int j=-2;j<=2;j++) 
      for(int i=-2;i<=2;i++)
        pixel_val += read_imagef(input, 2*c + make_int2(i, j), input_size) * w[i+2] * w[j+2];

    write_imagef(coarse, pos, coarse_size, pixel_val);
}

__device__ float laplacian(
    float* tex_coarse,  // coarse res gaussian
    float* tex_fine,    // fine res gaussian
    const int2 pos,                     // fine index
    const int2 clamped_pos,                    // clamped fine index
    const int2 fine_size,                    // fine size
    const int2 coarse_size)             // coarse size
{
    const float c = expand_gaussian(tex_coarse, clamped_pos, fine_size, coarse_size);
    return read_imagef(tex_fine, pos, fine_size) - c;
}

template<int num_gamma>
__global__ void laplacian_assemble(
    float* tex_input,    // original input buffer, gauss at current fine pyramid level
    float* tex_output1,  // state of reconstruction, coarse output buffer
    float* output0,      // reconstruction, one level finer, run kernel on this dimension
    const int2 fine_size)       // width and height of the fine buffers (l0)
{    
    const int2 pos = get_thread_pos();
    const int w = fine_size.x, h = fine_size.y;

    if(pos.x >= w || pos.y >= h) return;
    
    const int2 clamped_pos = clamp_boundary(pos, fine_size);
    const int2 coarse_size = make_int2((fine_size.x-1)/2+1, (fine_size.y-1)/2+1);
    
    float pixel_val = expand_gaussian(tex_output1, clamped_pos, fine_size, coarse_size);

    const float v = read_imagef(tex_input, pos, fine_size);
    int hi = 1;
    // what we mean is this:
    // for(;hi<num_gamma-1 && gamma[hi] <= v;hi++);
    for(;hi<num_gamma-1 && ((float)hi+.5f)/(float)num_gamma <= v;hi++);
    int lo = hi-1;
    // const float a = fminf(fmaxf((v - gamma[lo])/(gamma[hi]-gamma[lo]), 0.0f), 1.0f);
    const float a = fminf(fmaxf(v*num_gamma - ((float)lo+.5f), 0.0f), 1.0f);
    
    float l0 = laplacian(d_gamma_level1[lo], d_gamma_level0[lo], pos, clamped_pos, fine_size, coarse_size);
    float l1 = laplacian(d_gamma_level1[lo+1], d_gamma_level0[lo+1], pos, clamped_pos, fine_size, coarse_size);

    pixel_val += l0 * (1.0f-a) + l1 * a;
    write_imagef(output0, pos, fine_size, pixel_val);
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
    // midtone local contrast
    val += clarity * c * expf(-c*c/(2.0f*sigma*sigma/3.0f));
    return val;
}

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

    float pixel_val = read_imagef(tex_input, pos, size);
    pixel_val = curve(pixel_val, g, sigma, shadows, highlights, clarity);
    write_imagef(output, pos, size, pixel_val);
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

    float processed_val = 100.0f*read_imagef(tex_processed, pos + max_supp, processed_size);
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
        
        // Allocate buffers
        for(int l = 0; l < num_levels; l++)
        {
            const int level_width = dl(bwidth, l);
            const int level_height = dl(bheight, l);

            dev_padded.push_back(torch::empty({level_height, level_width}, torch::device(torch::kCUDA).dtype(torch::kFloat32)));
            dev_output.push_back(torch::empty({level_height, level_width}, torch::device(torch::kCUDA).dtype(torch::kFloat32)));

            for(int k = 0; k < num_gamma; k++) {
                auto tensor = torch::empty({level_height, level_width}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
                dev_processed[k].push_back(tensor);
            }
        }
    }

    // Helper functions to get buffer pointers
    float* processed_ptr(int level, int k) { return dev_processed[k][level].data_ptr<float>(); }
    float* padded_ptr(int level) { return dev_padded[level].data_ptr<float>(); }
    float* output_ptr(int level) { return dev_output[level].data_ptr<float>(); }

    // Processing function
    torch::Tensor process(const torch::Tensor& input) override
    {
        // Validate input tensor
        TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
        TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
        TORCH_CHECK(input.size(0) == height && input.size(1) == width, 
                    "Input tensor dimensions must match workspace dimensions");
        TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");

        auto stream = c10::cuda::getCurrentCUDAStream();
        CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

        // Create output tensor
        auto output = torch::empty({height, width}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        
        // Execute processing steps with error checking
        pad_input_step(input);
        build_gaussian_pyramid_step();
        process_gamma_curves_step();
        assemble_pyramid_step();
        write_back_step(output);

        // Final synchronization
        CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        
        return output;
    }
    
    // Helper functions for processing steps
    void pad_input_step(const torch::Tensor& input)
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        constexpr int block_size = 16;
        dim3 block(block_size, block_size);
        dim3 grid((bwidth + block_size - 1) / block_size, (bheight + block_size - 1) / block_size);
        
        pad_input<<<grid, block, 0, stream.stream()>>>(
            input.data_ptr<float>(), padded_ptr(0),
            make_int2(width, height), max_supp,
            make_int2(bwidth, bheight));
    }

    void build_gaussian_pyramid_step()
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        constexpr int block_size = 16;
        
        for(int l=1; l<num_levels; l++) {
            const int level_width = dl(bwidth, l);
            const int level_height = dl(bheight, l);
            
            dim3 block(block_size, block_size);
            dim3 grid((level_width + block_size - 1) / block_size, (level_height + block_size - 1) / block_size);
            
            gauss_reduce<<<grid, block, 0, stream.stream()>>>(
                padded_ptr(l-1), 
                (l == num_levels-1) ? output_ptr(l) : padded_ptr(l), 
                make_int2(level_width, level_height), make_int2(dl(bwidth, l-1), dl(bheight, l-1)));
        }
    }

    void process_gamma_curves_step()
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        constexpr int block_size = 16;
        
        for(int k=0; k<num_gamma; k++) {
            const float g = (k + 0.5f) / (float)num_gamma;
            
            dim3 block(block_size, block_size);
            dim3 grid((bwidth + block_size - 1) / block_size, (bheight + block_size - 1) / block_size);
            
            process_curve<<<grid, block, 0, stream.stream()>>>(
                padded_ptr(0), processed_ptr(0, k),
                g, sigma, shadows, highlights, clarity,
                make_int2(bwidth, bheight));

            // Create gaussian pyramids for this gamma level
            for(int l=1; l<num_levels; l++) {
                const int level_width = dl(bwidth, l);
                const int level_height = dl(bheight, l);
                
                dim3 block(block_size, block_size);
                dim3 grid((level_width + block_size - 1) / block_size, (level_height + block_size - 1) / block_size);
                
                gauss_reduce<<<grid, block, 0, stream.stream()>>>(
                    processed_ptr(l-1, k), processed_ptr(l, k), 
                    make_int2(level_width, level_height), make_int2(dl(bwidth, l-1), dl(bheight, l-1)));
            }
        }
    }

    void assemble_pyramid_step()
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        constexpr int block_size = 16;
        
        for(int l=num_levels-2; l >= 0; l--) {
            const int pw = dl(bwidth, l);
            const int ph = dl(bheight, l);
            
            dim3 block(block_size, block_size);
            dim3 grid((pw + block_size - 1) / block_size, (ph + block_size - 1) / block_size);
            
            // Prepare host arrays
            float* h_gamma_level0[max_gamma];
            float* h_gamma_level1[max_gamma];
            for(int k = 0; k < num_gamma; k++) {
                h_gamma_level0[k] = processed_ptr(l, k);
                h_gamma_level1[k] = processed_ptr(l+1, k);
            }
            
            // Copy to device symbols
            CUDA_CHECK(cudaMemcpyToSymbol(d_gamma_level0, h_gamma_level0, sizeof(float*) * max_gamma));
            CUDA_CHECK(cudaMemcpyToSymbol(d_gamma_level1, h_gamma_level1, sizeof(float*) * max_gamma));
            
            laplacian_assemble<num_gamma><<<grid, block, 0, stream.stream()>>>(
                padded_ptr(l), output_ptr(l+1), output_ptr(l),
                make_int2(pw, ph)); 
        }
    }

    void write_back_step(torch::Tensor& output)
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        constexpr int block_size = 16;
        dim3 final_block(block_size, block_size);
        dim3 final_grid((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);
        
        write_back<<<final_grid, final_block, 0, stream.stream()>>>(
            output_ptr(0), output.data_ptr<float>(),
            max_supp, make_int2(width, height), make_int2(bwidth, bheight));
    }
    
    // Parameter inspection
    py::dict get_parameters() const override {
        py::dict params;
        params["num_gamma"] = num_gamma;
        params["width"] = width;
        params["height"] = height;
        params["sigma"] = sigma;
        params["shadows"] = shadows;
        params["highlights"] = highlights;
        params["clarity"] = clarity;
        params["num_levels"] = num_levels;
        params["max_supp"] = max_supp;
        return params;
    }
    
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
        case 4:
            return std::make_shared<LaplacianImpl<4>>(device, width, height, sigma, shadows, highlights, clarity);
        case 6:
            return std::make_shared<LaplacianImpl<6>>(device, width, height, sigma, shadows, highlights, clarity);
        case 8:
            return std::make_shared<LaplacianImpl<8>>(device, width, height, sigma, shadows, highlights, clarity);
        default:
            throw std::runtime_error("Unsupported gamma count: " + std::to_string(num_gamma));
    }
}

