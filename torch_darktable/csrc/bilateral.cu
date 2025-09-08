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

#include "cuda_utils.h"
#include "device_math.h"
#include "bilateral.h"


// Helpers for grid indexing (x-fastest, then y, then z)
__device__ __forceinline__ int grid_index(int x, int y, int z, int sizex, int sizey, int sizez)
{
    return x + sizex * (y + sizey * z);
}

__device__ __forceinline__ float clampf(float v, float lo, float hi)
{
    return fminf(fmaxf(v, lo), hi);
}


// Zero out a 2D slice of the 3D grid interpreted as [sizex, sizey*sizez]
__global__ void zero_grid(float *grid, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    grid[x + width * y] = 0.0f;
}


// Splat kernel: builds the bilateral grid from luminance image
__global__ void splat_kernel(
    const float *in,
    float *grid,
    const int width,
    const int height,
    const int sizex,
    const int sizey,
    const int sizez,
    const float sigma_s,
    const float sigma_r)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    // Read luminance and map to grid coordinates
    const float L = in[y * width + x];
    const float3 g = make_float3(
        clampf(x / sigma_s, 0.0f, (float)(sizex - 1)),
        clampf(y / sigma_s, 0.0f, (float)(sizey - 1)),
        clampf(L / sigma_r, 0.0f, (float)(sizez - 1))
    );
    const int3 ib = make_int3(min((int)g.x, sizex - 2),
                              min((int)g.y, sizey - 2),
                              min((int)g.z, sizez - 2));
    const float3 f = make_float3(g.x - (float)ib.x,
                                 g.y - (float)ib.y,
                                 g.z - (float)ib.z);

    const int gi = grid_index(ib.x, ib.y, ib.z, sizex, sizey, sizez);

    // Contribution scale without extra 100x factor
    const float contrib = 1.0f / (sigma_s * sigma_s);

    // Trilinear atomic accumulation into 8 neighbors
    const float w000 = contrib * (1.0f - f.x) * (1.0f - f.y) * (1.0f - f.z);
    const float w100 = contrib * (       f.x) * (1.0f - f.y) * (1.0f - f.z);
    const float w010 = contrib * (1.0f - f.x) * (       f.y) * (1.0f - f.z);
    const float w110 = contrib * (       f.x) * (       f.y) * (1.0f - f.z);
    const float w001 = contrib * (1.0f - f.x) * (1.0f - f.y) * (       f.z);
    const float w101 = contrib * (       f.x) * (1.0f - f.y) * (       f.z);
    const float w011 = contrib * (1.0f - f.x) * (       f.y) * (       f.z);
    const float w111 = contrib * (       f.x) * (       f.y) * (       f.z);

    // Neighbor indices
    const int ox = 1;
    const int oy = sizex;
    const int oz = sizex * sizey;

    atomicAdd(grid + gi,                w000);
    atomicAdd(grid + gi + ox,           w100);
    atomicAdd(grid + gi + oy,           w010);
    atomicAdd(grid + gi + oy + ox,      w110);
    atomicAdd(grid + gi + oz,           w001);
    atomicAdd(grid + gi + oz + ox,      w101);
    atomicAdd(grid + gi + oz + oy,      w011);
    atomicAdd(grid + gi + oz + oy + ox, w111);
}


// Blur kernels, translated from bilateral.cl
__global__ void blur_line_kernel(
    const float *ibuf,
    float *obuf,
    const int offset1,
    const int offset2,
    const int offset3,
    const int size1,
    const int size2,
    const int size3)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(k >= size1 || j >= size2) return;

    const float w0 = 6.0f / 16.0f;
    const float w1 = 4.0f / 16.0f;
    const float w2 = 1.0f / 16.0f;

    int index = k * offset1 + j * offset2;

    float tmp1 = ibuf[index];
    obuf[index] = ibuf[index] * w0 + w1 * ibuf[index + offset3] + w2 * ibuf[index + 2 * offset3];
    index += offset3;
    float tmp2 = ibuf[index];
    obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp1) + w2 * ibuf[index + 2 * offset3];
    index += offset3;
    for(int i = 2; i < size3 - 2; i++)
    {
        const float tmp3 = ibuf[index];
        obuf[index] = ibuf[index] * w0
                    + w1 * (ibuf[index + offset3] + tmp2)
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
    const float *ibuf,
    float *obuf,
    const int offset1,
    const int offset2,
    const int offset3,
    const int size1,
    const int size2,
    const int size3)
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
    const float *in,
    const float *grid,
    float *out,
    const int width,
    const int height,
    const int sizex,
    const int sizey,
    const int sizez,
    const float sigma_s,
    const float sigma_r,
    const float detail)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    const float L = in[y * width + x];

    // Scale matches removal of 100x in splat: 100 * 0.04 = 4
    const float norm = -detail * sigma_r * 4.0f;

    const float3 g = make_float3(
        clampf(x / sigma_s, 0.0f, (float)(sizex - 1)),
        clampf(y / sigma_s, 0.0f, (float)(sizey - 1)),
        clampf(L / sigma_r, 0.0f, (float)(sizez - 1))
    );
    const int3 ib = make_int3(min((int)g.x, sizex - 2),
                              min((int)g.y, sizey - 2),
                              min((int)g.z, sizez - 2));
    const float3 f = make_float3(g.x - (float)ib.x,
                                 g.y - (float)ib.y,
                                 g.z - (float)ib.z);

    const int gi = grid_index(ib.x, ib.y, ib.z, sizex, sizey, sizez);
    const int ox = 1;
    const int oy = sizex;
    const int oz = sizex * sizey;

    const float Ldiff =
          grid[gi]               * (1.0f - f.x) * (1.0f - f.y) * (1.0f - f.z)
        + grid[gi + ox]          * (       f.x) * (1.0f - f.y) * (1.0f - f.z)
        + grid[gi + oy]          * (1.0f - f.x) * (       f.y) * (1.0f - f.z)
        + grid[gi + oy + ox]     * (       f.x) * (       f.y) * (1.0f - f.z)
        + grid[gi + oz]          * (1.0f - f.x) * (1.0f - f.y) * (       f.z)
        + grid[gi + oz + ox]     * (       f.x) * (1.0f - f.y) * (       f.z)
        + grid[gi + oz + oy]     * (1.0f - f.x) * (       f.y) * (       f.z)
        + grid[gi + oz + oy + ox]* (       f.x) * (       f.y) * (       f.z);

    const float Lout = fmaxf(0.0f, L + norm * Ldiff);
    out[y * width + x] = Lout;
}


// Bilateral workspace implementation (style similar to Laplacian)
struct BilateralImpl : public Bilateral
{
    torch::Device device_;
    int width, height;
    float sigma_s, sigma_r, detail;

    int sizex, sizey, sizez;

    torch::Tensor dev_grid;      // float32 [sizex*sizey*sizez]
    torch::Tensor dev_grid_tmp;  // float32 same size

    BilateralImpl(torch::Device device,
                  int width,
                  int height,
                  float sigma_s,
                  float sigma_r,
                  float detail)
        : device_(device), width(width), height(height),
          sigma_s(sigma_s), sigma_r(sigma_r), detail(detail)
    {
        TORCH_CHECK(width > 0 && height > 0, "Invalid dimensions");
        compute_grid_size();
        allocate_buffers();
    }

    void compute_grid_size()
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
        sigma_s = eff_sigma_s;
        sigma_r = eff_sigma_r;

        sizex = (int)ceilf(width  / sigma_s) + 1;
        sizey = (int)ceilf(height / sigma_s) + 1;
        sizez = (int)ceilf(L_range / sigma_r) + 1;
    }

    void allocate_buffers()
    {
        auto opts = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
        dev_grid = torch::empty({sizez, sizey, sizex}, opts);
        dev_grid_tmp = torch::empty({sizez, sizey, sizex}, opts);
    }

    torch::Tensor process(const torch::Tensor &luminance) override
    {
        TORCH_CHECK(luminance.dtype() == torch::kFloat32, "Input must be float32");
        TORCH_CHECK(luminance.dim() == 2, "Input must be 2D (H,W)");
        TORCH_CHECK(luminance.size(0) == height && luminance.size(1) == width,
                    "Input dims must match (H,W)");
        TORCH_CHECK(luminance.is_cuda(), "Input must be CUDA tensor");

        auto stream = c10::cuda::getCurrentCUDAStream();
        constexpr int block_size = 16;

        // Zero grid using torch
        dev_grid.zero_();

        // Splat
        {
            dim3 block(block_size, block_size);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            splat_kernel<<<grid, block, 0, stream.stream()>>>(
                luminance.data_ptr<float>(), dev_grid.data_ptr<float>(),
                width, height, sizex, sizey, sizez, sigma_s, sigma_r);
        }

        // Blur passes: two Gaussian and one derivative along Z
        {
            dim3 block(block_size, block_size);
            // Copy grid -> tmp via tensor op
            dev_grid_tmp.copy_(dev_grid);

            // 1) along X
            {
                dim3 grid_dim((sizez + block.x - 1) / block.x, (sizey + block.y - 1) / block.y);
                const int offset1 = sizex * sizey; // oz
                const int offset2 = sizex;         // oy
                const int offset3 = 1;             // ox
                blur_line_kernel<<<grid_dim, block, 0, stream.stream()>>>(
                    dev_grid_tmp.data_ptr<float>(), dev_grid.data_ptr<float>(),
                    offset1, offset2, offset3, sizez, sizey, sizex);
            }

            // 2) along Y
            {
                dim3 grid_dim((sizez + block.x - 1) / block.x, (sizex + block.y - 1) / block.y);
                const int offset1 = sizex * sizey; // oz
                const int offset2 = 1;             // ox
                const int offset3 = sizex;         // oy
                blur_line_kernel<<<grid_dim, block, 0, stream.stream()>>>(
                    dev_grid.data_ptr<float>(), dev_grid_tmp.data_ptr<float>(),
                    offset1, offset2, offset3, sizez, sizex, sizey);
            }

            // 3) derivative along Z
            {
                dim3 grid_dim((sizex + block.x - 1) / block.x, (sizey + block.y - 1) / block.y);
                const int offset1 = 1;             // ox
                const int offset2 = sizex;         // oy
                const int offset3 = sizex * sizey; // oz
                blur_line_z_kernel<<<grid_dim, block, 0, stream.stream()>>>(
                    dev_grid_tmp.data_ptr<float>(), dev_grid.data_ptr<float>(),
                    offset1, offset2, offset3, sizex, sizey, sizez);
            }
        }

        // Slice
        auto output = torch::empty({height, width}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        {
            dim3 block(block_size, block_size);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            slice_kernel<<<grid, block, 0, stream.stream()>>>(
                luminance.data_ptr<float>(), dev_grid.data_ptr<float>(), output.data_ptr<float>(),
                width, height, sizex, sizey, sizez, sigma_s, sigma_r, detail);
        }

        return output;
    }

    py::dict get_parameters() const override
    {
        py::dict p;
        p["width"] = width;
        p["height"] = height;
        p["sigma_s"] = sigma_s;
        p["sigma_r"] = sigma_r;
        p["detail"] = detail;
        p["sizex"] = sizex;
        p["sizey"] = sizey;
        p["sizez"] = sizez;
        return p;
    }

    void set_sigma_s(float v) override { sigma_s = v; compute_grid_size(); allocate_buffers(); }
    void set_sigma_r(float v) override { sigma_r = v; compute_grid_size(); allocate_buffers(); }
    void set_detail(float v) override { detail = v; }
};


std::shared_ptr<Bilateral> create_bilateral(
    torch::Device device,
    int width,
    int height,
    float sigma_s,
    float sigma_r,
    float detail)
{
    return std::make_shared<BilateralImpl>(device, width, height, sigma_s, sigma_r, detail);
}


