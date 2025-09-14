/*
    This file is part of darktable,
    copyright (c) 2025 darktable developers.

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
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include <algorithm>

#include "../cuda_utils.h"
#include "../device_math.h"
#include "bilateral.h"



// Trilinear interpolation from 3D grid at base index gi with fractional coords f
__device__ __forceinline__ float trilerp(const float *g, int gi, float fx, float fy, float fz, int ox, int oy, int oz)
{
    const float ax = 1.0f - fx, ay = 1.0f - fy, az = 1.0f - fz;
    return g[gi]                 * ax * ay * az
         + g[gi + ox]            * fx * ay * az
         + g[gi + oy]            * ax * fy * az
         + g[gi + oy + ox]       * fx * fy * az
         + g[gi + oz]            * ax * ay * fz
         + g[gi + oz + ox]       * fx * ay * fz
         + g[gi + oz + oy]       * ax * fy * fz
         + g[gi + oz + oy + ox]  * fx * fy * fz;
}

__device__ __forceinline__ void trilerp_add(
    float *g, int gi, float fx, float fy, float fz, int ox, int oy, int oz, float scale)
{
    const float ax = 1.0f - fx, ay = 1.0f - fy, az = 1.0f - fz;
    const float bx = fx,        by = fy,        bz = fz;

    atomicAdd(g + gi,                ax * ay * az * scale);
    atomicAdd(g + gi + ox,           bx * ay * az * scale);
    atomicAdd(g + gi + oy,           ax * by * az * scale);
    atomicAdd(g + gi + oy + ox,      bx * by * az * scale);
    atomicAdd(g + gi + oz,           ax * ay * bz * scale);
    atomicAdd(g + gi + oz + ox,      bx * ay * bz * scale);
    atomicAdd(g + gi + oz + oy,      ax * by * bz * scale);
    atomicAdd(g + gi + oz + oy + ox, bx * by * bz * scale);
}

struct GridSample
{
    int gi, ox, oy, oz;
    float fx, fy, fz;
};

__device__ __forceinline__ GridSample make_grid_sample(
    int x, int y, float L,
    int sizex, int sizey, int sizez,
    float sigma_s, float sigma_r)
{
    const float3 g = make_float3(
        clampf(x / sigma_s, 0.0f, (float)(sizex - 1)),
        clampf(y / sigma_s, 0.0f, (float)(sizey - 1)),
        clampf(L / sigma_r, 0.0f, (float)(sizez - 1)));
    const int3 ib = min(make_int3(g), make_int3(sizex - 2, sizey - 2, sizez - 2));
    const float3 f = g - ib;
    GridSample s;
    s.gi = grid_index(ib.x, ib.y, ib.z, sizex, sizey, sizez);
    s.ox = 1;
    s.oy = sizex;
    s.oz = sizex * sizey;
    s.fx = f.x; s.fy = f.y; s.fz = f.z;
    return s;
}


// Zero out a 2D slice of the 3D grid interpreted as [sizex, sizey*sizez]
__global__ void zero_grid(float *grid, int width, int height)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;
    grid[pos.x + width * pos.y] = 0.0f;
}


// Splat kernel: builds the bilateral grid from luminance image
__global__ void splat_kernel(
    const float *in, float *grid,
    const int width, const int height,
    const int sizex, const int sizey, const int sizez,
    const float sigma_s, const float sigma_r)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const float L = in[pos.y * width + pos.x];
    const GridSample s = make_grid_sample(pos.x, pos.y, L, sizex, sizey, sizez, sigma_s, sigma_r);
    const float contrib = 1.0f / (sigma_s * sigma_s);
    trilerp_add(grid, s.gi, s.fx, s.fy, s.fz, s.ox, s.oy, s.oz, contrib);
}

// Numerator splat: accumulate L * weights into grid
__global__ void splat_num_kernel(
    const float *in, float *grid,
    const int width, const int height,
    const int sizex, const int sizey, const int sizez,
    const float sigma_s, const float sigma_r)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const float L = in[pos.y * width + pos.x];
    const GridSample s = make_grid_sample(pos.x, pos.y, L, sizex, sizey, sizez, sigma_s, sigma_r);
    const float contrib = 1.0f / (sigma_s * sigma_s);
    trilerp_add(grid, s.gi, s.fx, s.fy, s.fz, s.ox, s.oy, s.oz, contrib * L);
}


// Blur kernels, translated from bilateral.cl
__global__ void blur_line_kernel(
    const float *ibuf, float *obuf,
    const int offset1, const int offset2, const int offset3,
    const int size1, const int size2, const int size3)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(k >= size1 || j >= size2) return;

    const float w0 = 6.0f / 16.0f;
    const float w1 = 4.0f / 16.0f;
    const float w2 = 1.0f / 16.0f;

    int index = k * offset1 + j * offset2;

    float tmp1 = ibuf[index];
    obuf[index] = ibuf[index] * w0 + w1 * ibuf[index + offset3] 
                + w2 * ibuf[index + 2 * offset3];
    index += offset3;
    float tmp2 = ibuf[index];
    obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp1) 
                + w2 * ibuf[index + 2 * offset3];
    index += offset3;
    for(int i = 2; i < size3 - 2; i++)
    {
        const float tmp3 = ibuf[index];
        obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp2)
                    + w2 * (ibuf[index + 2 * offset3] + tmp1);
        index += offset3;
        tmp1 = tmp2;
        tmp2 = tmp3;
    }
    const float tmp3 = ibuf[index];
    obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp2) + w2 * tmp1;
    index += offset3;
    obuf[index] = ibuf[index] * w0 + w1 * tmp3 + w2 * tmp2;
}


__global__ void blur_line_z_kernel(
    const float *ibuf, float *obuf,
    const int offset1, const int offset2, const int offset3,
    const int size1, const int size2, const int size3)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(k >= size1 || j >= size2) return;

    const float w1 = 4.0f / 16.0f;
    const float w2 = 2.0f / 16.0f;

    int index = k * offset1 + j * offset2;

    float tmp1 = ibuf[index];
    obuf[index] = w1 * ibuf[index + offset3] + w2 * ibuf[index + 2 * offset3];
    index += offset3;
    float tmp2 = ibuf[index];
    obuf[index] = w1 * (ibuf[index + offset3] - tmp1) + w2 * ibuf[index + 2 * offset3];
    index += offset3;
    for(int i = 2; i < size3 - 2; i++)
    {
        const float tmp3 = ibuf[index];
        obuf[index] = + w1 * (ibuf[index + offset3]   - tmp2)
                      + w2 * (ibuf[index + 2 * offset3] - tmp1);
        index += offset3;
        tmp1 = tmp2;
        tmp2 = tmp3;
    }
    const float tmp3 = ibuf[index];
    obuf[index] = w1 * (ibuf[index + offset3] - tmp2) - w2 * tmp1;
    index += offset3;
    obuf[index] = - w1 * tmp3 - w2 * tmp2;
}


// Slice kernel: outputs processed luminance
__global__ void slice_kernel(
    const float *in, const float *grid, float *out,
    const int width, const int height,
    const int sizex, const int sizey, const int sizez,
    const float sigma_s, const float sigma_r,
    const float detail)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const float L = in[pos.y * width + pos.x];

    // Scale matches removal of 100x in splat: 100 * 0.04 = 4
    const float norm = -detail * sigma_r * 4.0f;

    const GridSample s = make_grid_sample(pos.x, pos.y, L, sizex, sizey, sizez, sigma_s, sigma_r);
    const float Ldiff = trilerp(grid, s.gi, s.fx, s.fy, s.fz, s.ox, s.oy, s.oz);

    const float Lout = fmaxf(0.0f, L + norm * Ldiff);
    out[pos.y * width + pos.x] = Lout;
}

// Denoise: slice weighted average from two blurred grids (sum_wL, sum_w)
__global__ void slice_denoise_kernel(
    const float *in, const float *grid_num, const float *grid_den, float *out,
    const int width, const int height,
    const int sizex, const int sizey, const int sizez,
    const float sigma_s, const float sigma_r, const float amount)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const float L = in[pos.y * width + pos.x];
    const GridSample s = make_grid_sample(pos.x, pos.y, L, sizex, sizey, sizez, sigma_s, sigma_r);
    const float num = trilerp(grid_num, s.gi, s.fx, s.fy, s.fz, s.ox, s.oy, s.oz);
    const float den = trilerp(grid_den, s.gi, s.fx, s.fy, s.fz, s.ox, s.oy, s.oz);

    const float denoised = (den > 1e-8f) ? (num / den) : L;
    const float Lout = (1.0f - amount) * L + amount * denoised;
    out[pos.y * width + pos.x] = Lout;
}


// Bilateral workspace implementation (style similar to Laplacian)
struct BilateralImpl : public Bilateral
{
    torch::Device device_;
    int width, height;
    float sigma_s, sigma_r;

    // Cache tensors for efficiency; infer grid dims from current parameters when (re)allocating
    torch::Tensor dev_grid;
    torch::Tensor dev_grid_tmp;
    torch::Tensor dev_grid_num; // reserved
    torch::Tensor dev_grid_den; // reserved

    BilateralImpl(torch::Device device,
                  int width, int height,
                  float sigma_s, float sigma_r)
        : device_(device), width(width), height(height),
          sigma_s(sigma_s), sigma_r(sigma_r)
    {
        TORCH_CHECK(width > 0 && height > 0, "Invalid dimensions");
    }

    std::tuple<int,int,int> compute_grid_size() const
    {
        float ss = sigma_s;
        if(ss < 0.5f) ss = 0.5f;

        // L range assumes input luminance in [0,1]
        const float L_range = 1.0f;

        float gx = roundf(width  / ss);
        float gy = roundf(height / ss);
        float gz = roundf(L_range / sigma_r);
        gx = fminf(fmaxf(gx, 4.0f), 3000.0f);
        gy = fminf(fmaxf(gy, 4.0f), 3000.0f);
        gz = fminf(fmaxf(gz, 4.0f), 50.0f);

        // Effective sigmas after potential clamping
        const float eff_sigma_s = fmaxf(height / gy, width / gx);
        const float eff_sigma_r = L_range / gz;
        const float s_s = eff_sigma_s;
        const float s_r = eff_sigma_r;
        const int sx = (int)ceilf(width  / s_s) + 1;
        const int sy = (int)ceilf(height / s_s) + 1;
        const int sz = (int)ceilf(L_range / s_r) + 1;
        return {sx, sy, sz};
    }

    void invalidate_buffers()
    {
        dev_grid = torch::Tensor();
        dev_grid_tmp = torch::Tensor();
        dev_grid_num = torch::Tensor();
        dev_grid_den = torch::Tensor();
    }

    void check_input_image(const torch::Tensor &luminance) const
    {
        TORCH_CHECK(luminance.dtype() == torch::kFloat32, "Input must be float32");
        TORCH_CHECK(luminance.dim() == 2, "Input must be 2D (H,W)");
        TORCH_CHECK(luminance.size(0) == height && luminance.size(1) == width,
                    "Input dims must match (H,W)");
        TORCH_CHECK(luminance.is_cuda(), "Input must be CUDA tensor");
    }

    // Host helpers for blur scheduling
    void blur_xy(torch::Tensor &src, torch::Tensor &tmp, torch::Tensor &y_out, 
                 int sx, int sy, int sz, cudaStream_t stream)
    {
        blur_line_kernel<<<grid2d(sz, sy), block_size_2d, 0, stream>>>(
            src.data_ptr<float>(), tmp.data_ptr<float>(), sx * sy, sx, 1, sz, sy, sx);
        blur_line_kernel<<<grid2d(sz, sx), block_size_2d, 0, stream>>>(
            tmp.data_ptr<float>(), y_out.data_ptr<float>(), sx * sy, 1, sx, sz, sx, sy);
    }

    void blur_z_gaussian(torch::Tensor &y_in, torch::Tensor &z_out, 
                         int sx, int sy, int sz, cudaStream_t stream)
    {
        blur_line_kernel<<<grid2d(sx, sy), block_size_2d, 0, stream>>>(
            y_in.data_ptr<float>(), z_out.data_ptr<float>(), 1, sx, sx * sy, sx, sy, sz);
    }

    void blur_z_derivative(torch::Tensor &y_in, torch::Tensor &z_out, 
                           int sx, int sy, int sz, cudaStream_t stream)
    {
        blur_line_z_kernel<<<grid2d(sx, sy), block_size_2d, 0, stream>>>(
            y_in.data_ptr<float>(), z_out.data_ptr<float>(), 1, sx, sx * sy, sx, sy, sz);
    }

    // Ensure buffers for detail mode (single grid + tmp)
    void ensure_detail_buffers(int sx, int sy, int sz)
    {
        auto opts = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
        if (!dev_grid.defined()) {
            dev_grid = torch::empty({sz, sy, sx}, opts);
            dev_grid_tmp = torch::empty({sz, sy, sx}, opts);
        }
    }

    // removed denoise buffers

    

    torch::Tensor process(const torch::Tensor &luminance, float detail) override
    {
        check_input_image(luminance);

        auto stream = c10::cuda::getCurrentCUDAStream();

        // Compute grid sizes on-the-fly and (re)allocate cached buffers if needed
        auto [sx, sy, sz] = compute_grid_size();
        ensure_detail_buffers(sx, sy, sz);
        dev_grid.zero_();

  
        splat_kernel<<<grid2d(width, height), block_size_2d, 0, stream.stream()>>>(
            luminance.data_ptr<float>(), dev_grid.data_ptr<float>(),
            width, height, sx, sy, sz, sigma_s, sigma_r);
  
        // Blur passes via helpers
        blur_xy(dev_grid, dev_grid_tmp, dev_grid, sx, sy, sz, stream.stream());
        blur_z_derivative(dev_grid, dev_grid_tmp, sx, sy, sz, stream.stream());

        // Slice
        auto output = torch::empty({height, width}, luminance.options());
        slice_kernel<<<grid2d(width, height), block_size_2d, 0, stream.stream()>>>(
            luminance.data_ptr<float>(), dev_grid_tmp.data_ptr<float>(), output.data_ptr<float>(),
            width, height, sx, sy, sz, sigma_s, sigma_r, detail);

        return output;
    }

    

    void set_sigma_s(float v) override { sigma_s = v; invalidate_buffers(); }
    void set_sigma_r(float v) override { sigma_r = v; invalidate_buffers(); }
    float get_sigma_s() const override { return sigma_s; }
    float get_sigma_r() const override { return sigma_r; }
};


std::shared_ptr<Bilateral> create_bilateral(
    torch::Device device,
    int width, int height,
    float sigma_s, float sigma_r) {
    return std::make_shared<BilateralImpl>(
      device, width, height, sigma_s, sigma_r);
}


