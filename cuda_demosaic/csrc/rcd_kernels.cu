/*
 * RCD Demosaic CUDA Kernels
 * Converted from OpenCL kernels for better PyTorch integration
 * Based on darktable's RCD (Ratio Corrected Demosaicing) algorithm
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

// FC macro for Bayer pattern (from darktable)
#define FC(row, col, filters) ((filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1)) & 3)

// Forward declaration of shared function from ppg_kernels.cu
__global__ void border_interpolate_kernel(float* input, float4* output, int width, int height, uint32_t filters, int border);

// Helper functions
__device__ __forceinline__ float fsquare(float x) {
    return x * x;
}

__device__ __forceinline__ float clipf(float x) {
    return fmaxf(fminf(x, 1.0f), 0.0f);
}

__device__ __forceinline__ float dt_fast_expf(float x) {
    return expf(x);
}

__device__ __forceinline__ float dt_fast_hypot(float x, float y) {
    return hypotf(x, y);
}

__device__ __forceinline__ float dtcl_sqrt(float x) {
    return sqrtf(x);
}

// Mix function equivalent
__device__ __forceinline__ float mix(float a, float b, float t) {
    return (1.0f - t) * a + t * b;
}

static __device__ __forceinline__ float calcBlendFactor(float val, float threshold)
{
    // sigmoid function
    // result is in ]0;1] range
    // inflexion point is at (x, y) (threshold, 0.5)
    return 1.0f / (1.0f + dt_fast_expf(16.0f - (16.0f / threshold) * val));
}

// Populate cfa and rgb data by normalized input
__global__ void rcd_populate_kernel(float* input, float* cfa, float* rgb0, float* rgb1, float* rgb2, 
                                   int width, int height, uint32_t filters, float scale)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col >= width || row >= height) return;
    
    const float val = scale * fmaxf(0.0f, input[row * width + col]);
    const int color = FC(row, col, filters);

    float* rgbcol = rgb0;
    if(color == 1) rgbcol = rgb1;
    else if(color == 2) rgbcol = rgb2;

    const int idx = row * width + col;
    cfa[idx] = rgbcol[idx] = val;
}

// Write back-normalized data in rgb channels to output
__global__ void rcd_write_output_kernel(float4* output, float* rgb0, float* rgb1, float* rgb2, 
                                       int width, int height, float scale, int border)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(!(col >= border && col < width - border && row >= border && row < height - border)) return;
    
    const int idx = row * width + col;
    output[idx] = make_float4(fmaxf(scale * rgb0[idx], 0.0f), 
                             fmaxf(scale * rgb1[idx], 0.0f), 
                             fmaxf(scale * rgb2[idx], 0.0f), 
                             1.0f);
}

// Step 1.1: Calculate a squared vertical and horizontal high pass filter on color differences
__global__ void rcd_step_1_1_kernel(float* cfa, float* v_diff, float* h_diff, int width, int height)
{
    const int col = 3 + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = 3 + blockIdx.y * blockDim.y + threadIdx.y;
    if((row > height - 4) || (col > width - 4)) return;
    
    const int idx = row * width + col;
    const int w2 = 2 * width;
    const int w3 = 3 * width;

    v_diff[idx] = fsquare(cfa[idx - w3] - 3.0f * cfa[idx - w2] - cfa[idx - width] + 6.0f * cfa[idx] - cfa[idx + width] - 3.0f * cfa[idx + w2] + cfa[idx + w3]);
    h_diff[idx] = fsquare(cfa[idx - 3] - 3.0f * cfa[idx - 2] - cfa[idx - 1] + 6.0f * cfa[idx] - cfa[idx + 1] - 3.0f * cfa[idx + 2] + cfa[idx + 3]);
}

// Step 1.2: Calculate vertical and horizontal local discrimination
__global__ void rcd_step_1_2_kernel(float* VH_dir, float* v_diff, float* h_diff, int width, int height)
{
    const int col = 2 + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = 2 + blockIdx.y * blockDim.y + threadIdx.y;
    if((row > height - 3) || (col > width - 3)) return;
    
    const int idx = row * width + col;
    const float eps = 1e-10f;

    const float V_Stat = fmaxf(eps, v_diff[idx - width] + v_diff[idx] + v_diff[idx + width]);
    const float H_Stat = fmaxf(eps, h_diff[idx - 1] + h_diff[idx] + h_diff[idx + 1]);
    VH_dir[idx] = V_Stat / (V_Stat + H_Stat);
}

// Step 2.1: Low pass filter incorporating green, red and blue local samples from the raw data
__global__ void rcd_step_2_1_kernel(float* lpf, float* cfa, int width, int height, uint32_t filters)
{
    const int row = 2 + blockIdx.y * blockDim.y + threadIdx.y;
    const int col = 2 + (FC(row, 0, filters) & 1) + 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if((col > width - 2) || (row > height - 2)) return;
    
    const int idx = row * width + col;

    lpf[idx / 2] = cfa[idx]
       + 0.5f * (cfa[idx - width] + cfa[idx + width] + cfa[idx - 1] + cfa[idx + 1])
      + 0.25f * (cfa[idx - width - 1] + cfa[idx - width + 1] + cfa[idx + width - 1] + cfa[idx + width + 1]);
}

// Step 3.1: Populate the green channel at blue and red CFA positions
__global__ void rcd_step_3_1_kernel(float* lpf, float* cfa, float* rgb1, float* VH_Dir, 
                                   int width, int height, uint32_t filters)
{
    const int row = 4 + blockIdx.y * blockDim.y + threadIdx.y;
    const int col = 4 + (FC(row, 0, filters) & 1) + 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if((col > width - 5) || (row > height - 5)) return;
    
    const int idx = row * width + col;
    const int lidx = idx / 2;
    const int w2 = 2 * width;
    const int w3 = 3 * width;
    const int w4 = 4 * width;
    const float eps = 1e-5f;

    // Refined vertical and horizontal local discrimination
    const float VH_Central_Value = VH_Dir[idx];
    const float VH_Neighbourhood_Value = 0.25f * (VH_Dir[idx - width - 1] + VH_Dir[idx - width + 1] + VH_Dir[idx + width - 1] + VH_Dir[idx + width + 1]);
    const float VH_Disc = (fabsf(0.5f - VH_Central_Value) < fabsf(0.5f - VH_Neighbourhood_Value)) ? VH_Neighbourhood_Value : VH_Central_Value;

    const float cfai = cfa[idx];
    // Cardinal gradients
    const float N_Grad = eps + fabsf(cfa[idx - width] - cfa[idx + width]) + fabsf(cfai - cfa[idx - w2]) + fabsf(cfa[idx - width] - cfa[idx - w3]) + fabsf(cfa[idx - w2] - cfa[idx - w4]);
    const float S_Grad = eps + fabsf(cfa[idx + width] - cfa[idx - width]) + fabsf(cfai - cfa[idx + w2]) + fabsf(cfa[idx + width] - cfa[idx + w3]) + fabsf(cfa[idx + w2] - cfa[idx + w4]);
    const float W_Grad = eps + fabsf(cfa[idx - 1] - cfa[idx + 1]) + fabsf(cfai - cfa[idx - 2]) + fabsf(cfa[idx - 1] - cfa[idx - 3]) + fabsf(cfa[idx - 2] - cfa[idx - 4]);
    const float E_Grad = eps + fabsf(cfa[idx + 1] - cfa[idx - 1]) + fabsf(cfai - cfa[idx + 2]) + fabsf(cfa[idx + 1] - cfa[idx + 3]) + fabsf(cfa[idx + 2] - cfa[idx + 4]);

    const float lfpi = lpf[lidx];
    // Cardinal pixel estimations
    const float N_Est = cfa[idx - width] * (lfpi + lfpi) / (eps + lfpi + lpf[lidx - width]);
    const float S_Est = cfa[idx + width] * (lfpi + lfpi) / (eps + lfpi + lpf[lidx + width]);
    const float W_Est = cfa[idx - 1] * (lfpi + lfpi) / (eps + lfpi + lpf[lidx - 1]);
    const float E_Est = cfa[idx + 1] * (lfpi + lfpi) / (eps + lfpi + lpf[lidx + 1]);

    // Vertical and horizontal estimations
    const float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
    const float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);

    // G@B and G@R interpolation
    rgb1[idx] = mix(V_Est, H_Est, VH_Disc);
}

// Step 4.1: Calculate the square of the P/Q diagonals color difference high pass filter
__global__ void rcd_step_4_1_kernel(float* cfa, float* p_diff, float* q_diff, 
                                   int width, int height, uint32_t filters)
{
    const int row = 3 + blockIdx.y * blockDim.y + threadIdx.y;
    const int col = 3 + 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if((col > width - 4) || (row > height - 4)) return;
    
    const int idx = row * width + col;
    const int idx2 = idx / 2;
    const int w2 = 2 * width;
    const int w3 = 3 * width;

    p_diff[idx2] = fsquare((cfa[idx - w3 - 3] - cfa[idx - width - 1] - cfa[idx + width + 1] + cfa[idx + w3 + 3]) - 3.0f * (cfa[idx - w2 - 2] + cfa[idx + w2 + 2]) + 6.0f * cfa[idx]);
    q_diff[idx2] = fsquare((cfa[idx - w3 + 3] - cfa[idx - width + 1] - cfa[idx + width - 1] + cfa[idx + w3 - 3]) - 3.0f * (cfa[idx - w2 + 2] + cfa[idx + w2 - 2]) + 6.0f * cfa[idx]);
}

// Step 4.2: Calculate P/Q diagonals local discrimination strength
__global__ void rcd_step_4_2_kernel(float* PQ_dir, float* p_diff, float* q_diff, 
                                   int width, int height, uint32_t filters)
{
    const int row = 2 + blockIdx.y * blockDim.y + threadIdx.y;
    const int col = 2 + (FC(row, 0, filters) & 1) + 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if((col > width - 3) || (row > height - 3)) return;
    
    const int idx = row * width + col;
    const int idx2 = idx / 2;
    const int idx3 = (idx - width - 1) / 2;
    const int idx4 = (idx + width - 1) / 2;
    const float eps = 1e-10f;

    const float P_Stat = fmaxf(eps, p_diff[idx3] + p_diff[idx2] + p_diff[idx4 + 1]);
    const float Q_Stat = fmaxf(eps, q_diff[idx3 + 1] + q_diff[idx2] + q_diff[idx4]);
    PQ_dir[idx2] = P_Stat / (P_Stat + Q_Stat);
}

// Step 5.1: Populate the red and blue channels at blue and red CFA positions
__global__ void rcd_step_5_1_kernel(float* PQ_dir, float* rgb0, float* rgb1, float* rgb2, 
                                   int width, int height, uint32_t filters)
{
    const int row = 4 + blockIdx.y * blockDim.y + threadIdx.y;
    const int col = 4 + (FC(row, 0, filters) & 1) + 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if((col > width - 4) || (row > height - 4)) return;

    const int color = 2 - FC(row, col, filters);

    float* rgbc = rgb0;
    if(color == 1) rgbc = rgb1;
    else if(color == 2) rgbc = rgb2;

    const int idx = row * width + col;
    const int pqidx = idx / 2;
    const int pqidx2 = (idx - width - 1) / 2;
    const int pqidx3 = (idx + width - 1) / 2;
    const int w2 = 2 * width;
    const int w3 = 3 * width;
    const float eps = 1e-5f;

    const float PQ_Central_Value = PQ_dir[pqidx];
    const float PQ_Neighbourhood_Value = 0.25f * (PQ_dir[pqidx2] + PQ_dir[pqidx2 + 1] + PQ_dir[pqidx3] + PQ_dir[pqidx3 + 1]);
    const float PQ_Disc = (fabsf(0.5f - PQ_Central_Value) < fabsf(0.5f - PQ_Neighbourhood_Value)) ? PQ_Neighbourhood_Value : PQ_Central_Value;

    const float NW_Grad = eps + fabsf(rgbc[idx - width - 1] - rgbc[idx + width + 1]) + fabsf(rgbc[idx - width - 1] - rgbc[idx - w3 - 3]) + fabsf(rgb1[idx] - rgb1[idx - w2 - 2]);
    const float NE_Grad = eps + fabsf(rgbc[idx - width + 1] - rgbc[idx + width - 1]) + fabsf(rgbc[idx - width + 1] - rgbc[idx - w3 + 3]) + fabsf(rgb1[idx] - rgb1[idx - w2 + 2]);
    const float SW_Grad = eps + fabsf(rgbc[idx - width + 1] - rgbc[idx + width - 1]) + fabsf(rgbc[idx + width - 1] - rgbc[idx + w3 - 3]) + fabsf(rgb1[idx] - rgb1[idx + w2 - 2]);
    const float SE_Grad = eps + fabsf(rgbc[idx - width - 1] - rgbc[idx + width + 1]) + fabsf(rgbc[idx + width + 1] - rgbc[idx + w3 + 3]) + fabsf(rgb1[idx] - rgb1[idx + w2 + 2]);

    const float NW_Est = rgbc[idx - width - 1] - rgb1[idx - width - 1];
    const float NE_Est = rgbc[idx - width + 1] - rgb1[idx - width + 1];
    const float SW_Est = rgbc[idx + width - 1] - rgb1[idx + width - 1];
    const float SE_Est = rgbc[idx + width + 1] - rgb1[idx + width + 1];

    const float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad);
    const float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad);

    rgbc[idx] = rgb1[idx] + mix(P_Est, Q_Est, PQ_Disc);
}

// Step 5.2: Populate the red and blue channels at green CFA positions
__global__ void rcd_step_5_2_kernel(float* VH_dir, float* rgb0, float* rgb1, float* rgb2, 
                                   int width, int height, uint32_t filters)
{
    const int row = 4 + blockIdx.y * blockDim.y + threadIdx.y;
    const int col = 4 + (FC(row, 1, filters) & 1) + 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if((col > width - 4) || (row > height - 4)) return;

    const int idx = row * width + col;
    const int w2 = 2 * width;
    const int w3 = 3 * width;
    const float eps = 1e-5f;

    // Refined vertical and horizontal local discrimination
    const float VH_Central_Value = VH_dir[idx];
    const float VH_Neighbourhood_Value = 0.25f * (VH_dir[idx - width - 1] + VH_dir[idx - width + 1] + VH_dir[idx + width - 1] + VH_dir[idx + width + 1]);
    const float VH_Disc = (fabsf(0.5f - VH_Central_Value) < fabsf(0.5f - VH_Neighbourhood_Value)) ? VH_Neighbourhood_Value : VH_Central_Value;

    const float rgbi1 = rgb1[idx];
    const float N1 = eps + fabsf(rgbi1 - rgb1[idx - w2]);
    const float S1 = eps + fabsf(rgbi1 - rgb1[idx + w2]);
    const float W1 = eps + fabsf(rgbi1 - rgb1[idx - 2]);
    const float E1 = eps + fabsf(rgbi1 - rgb1[idx + 2]);

    const float rgb1mw1 = rgb1[idx - width];
    const float rgb1pw1 = rgb1[idx + width];
    const float rgb1m1 = rgb1[idx - 1];
    const float rgb1p1 = rgb1[idx + 1];

    for(int c = 0; c <= 2; c += 2)
    {
        float* rgbc = (c == 0) ? rgb0 : rgb2;

        const float SNabs = fabsf(rgbc[idx - width] - rgbc[idx + width]);
        const float EWabs = fabsf(rgbc[idx - 1] - rgbc[idx + 1]);

        // Cardinal gradients
        const float N_Grad = N1 + SNabs + fabsf(rgbc[idx - width] - rgbc[idx - w3]);
        const float S_Grad = S1 + SNabs + fabsf(rgbc[idx + width] - rgbc[idx + w3]);
        const float W_Grad = W1 + EWabs + fabsf(rgbc[idx - 1] - rgbc[idx - 3]);
        const float E_Grad = E1 + EWabs + fabsf(rgbc[idx + 1] - rgbc[idx + 3]);

        // Cardinal colour differences
        const float N_Est = rgbc[idx - width] - rgb1mw1;
        const float S_Est = rgbc[idx + width] - rgb1pw1;
        const float W_Est = rgbc[idx - 1] - rgb1m1;
        const float E_Est = rgbc[idx + 1] - rgb1p1;

        // Vertical and horizontal estimations
        const float V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad);
        const float H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad);

        // R@G and B@G interpolation
        rgbc[idx] = rgb1[idx] + mix(V_Est, H_Est, VH_Disc);
    }
}

// Border interpolation kernels
__global__ void rcd_border_green_kernel(float* input, float4* output, int width, int height,
                                       uint32_t filters, int border)
{
    extern __shared__ float green_buffer[];
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
    // cells of 1*float per pixel with a surrounding border of 3 cells
    const int stride = xlsz + 2*3;
    const int maxbuf = stride * (ylsz + 2*3);

    // coordinates of top left pixel of buffer
    // this is 3 pixel left and above of the work group origin
    const int xul = xgid * xlsz - 3;
    const int yul = ygid * ylsz - 3;

    // populate local memory buffer
    for(int n = 0; n <= maxbuf/lsz; n++)
    {
        const int bufidx = n * lsz + l;
        if(bufidx >= maxbuf) continue;
        const int xx = xul + bufidx % stride;
        const int yy = yul + bufidx / stride;
        green_buffer[bufidx] = (xx >= 0 && yy >= 0 && xx < width && yy < height) ? fmaxf(0.0f, input[yy * width + xx]) : 0.0f;
    }

    // center buffer around current x,y-Pixel
    float* centered_buffer = green_buffer + (ylid + 3) * stride + xlid + 3;

    __syncthreads();

    if(x >= width - 3 || x < 3 || y >= height - 3 || y < 3) return;
    if(x >= border && x < width - border && y >= border && y < height - border) return;

    // process all non-green pixels
    const int row = y;
    const int col = x;
    const int c = FC(row, col, filters);
    float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // output color

    const float pc = centered_buffer[0];

    if     (c == 0) color.x = pc; // red
    else if(c == 1) color.y = pc; // green1
    else if(c == 2) color.z = pc; // blue
    else            color.y = pc; // green2

    // fill green layer for red and blue pixels:
    if(c == 0 || c == 2)
    {
        // look up horizontal and vertical neighbours, sharpened weight:
        const float pym  = centered_buffer[-1 * stride];
        const float pym2 = centered_buffer[-2 * stride];
        const float pym3 = centered_buffer[-3 * stride];
        const float pyM  = centered_buffer[ 1 * stride];
        const float pyM2 = centered_buffer[ 2 * stride];
        const float pyM3 = centered_buffer[ 3 * stride];
        const float pxm  = centered_buffer[-1];
        const float pxm2 = centered_buffer[-2];
        const float pxm3 = centered_buffer[-3];
        const float pxM  = centered_buffer[ 1];
        const float pxM2 = centered_buffer[ 2];
        const float pxM3 = centered_buffer[ 3];
        const float guessx = (pxm + pc + pxM) * 2.0f - pxM2 - pxm2;
        const float diffx  = (fabsf(pxm2 - pc) +
                              fabsf(pxM2 - pc) +
                              fabsf(pxm  - pxM)) * 3.0f +
                             (fabsf(pxM3 - pxM) + fabsf(pxm3 - pxm)) * 2.0f;
        const float guessy = (pym + pc + pyM) * 2.0f - pyM2 - pym2;
        const float diffy  = (fabsf(pym2 - pc) +
                              fabsf(pyM2 - pc) +
                              fabsf(pym  - pyM)) * 3.0f +
                             (fabsf(pyM3 - pyM) + fabsf(pym3 - pym)) * 2.0f;
        if(diffx > diffy)
        {
            // use guessy
            const float m = fminf(pym, pyM);
            const float M = fmaxf(pym, pyM);
            color.y = fmaxf(fminf(guessy*0.25f, M), m);
        }
        else
        {
            const float m = fminf(pxm, pxM);
            const float M = fmaxf(pxm, pxM);
            color.y = fmaxf(fminf(guessx*0.25f, M), m);
        }
    }
    output[y * width + x] = make_float4(fmaxf(color.x, 0.0f), fmaxf(color.y, 0.0f), fmaxf(color.z, 0.0f), 1.0f);
}

__global__ void rcd_border_redblue_kernel(float4* input, float4* output, int width, int height,
                                         uint32_t filters, int border)
{
    extern __shared__ float4 redblue_buffer[];
    // image in contains full green and sparse r b
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
    // cells of float4 per pixel with a surrounding border of 1 cell
    const int stride = xlsz + 2;
    const int maxbuf = stride * (ylsz + 2);

    // coordinates of top left pixel of buffer
    // this is 1 pixel left and above of the work group origin
    const int xul = xgid * xlsz - 1;
    const int yul = ygid * ylsz - 1;

    // populate local memory buffer
    for(int n = 0; n <= maxbuf/lsz; n++)
    {
        const int bufidx = n * lsz + l;
        if(bufidx >= maxbuf) continue;
        const int xx = xul + bufidx % stride;
        const int yy = yul + bufidx / stride;
        redblue_buffer[bufidx] = (xx >= 0 && yy >= 0 && xx < width && yy < height) ? 
                        make_float4(fmaxf(0.0f, input[yy * width + xx].x), 
                                   fmaxf(0.0f, input[yy * width + xx].y),
                                   fmaxf(0.0f, input[yy * width + xx].z), 1.0f) : 
                        make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // center buffer around current x,y-Pixel
    float4* centered_buffer = redblue_buffer + (ylid + 1) * stride + xlid + 1;

    __syncthreads();

    if(x >= width || y >= height) return;
    if(x >= border && x < width - border && y >= border && y < height - border) return;

    const int row = y;
    const int col = x;
    const int c = FC(row, col, filters);
    float4 color = centered_buffer[0];
    if(row > 0 && col > 0 && col < width - 1 && row < height - 1)
    {
        if(c == 1 || c == 3)
        { // calculate red and blue for green pixels:
            // need 4-nbhood:
            const float4 nt = centered_buffer[-stride];
            const float4 nb = centered_buffer[ stride];
            const float4 nl = centered_buffer[-1];
            const float4 nr = centered_buffer[ 1];
            if(FC(row, col+1, filters) == 0) // red nb in same row
            {
                color.z = (nt.z + nb.z + 2.0f*color.y - nt.y - nb.y)*0.5f;
                color.x = (nl.x + nr.x + 2.0f*color.y - nl.y - nr.y)*0.5f;
            }
            else
            { // blue nb
                color.x = (nt.x + nb.x + 2.0f*color.y - nt.y - nb.y)*0.5f;
                color.z = (nl.z + nr.z + 2.0f*color.y - nl.y - nr.y)*0.5f;
            }
        }
        else
        {
            // get 4-star-nbhood:
            const float4 ntl = centered_buffer[-stride - 1];
            const float4 ntr = centered_buffer[-stride + 1];
            const float4 nbl = centered_buffer[ stride - 1];
            const float4 nbr = centered_buffer[ stride + 1];

            if(c == 0)
            { // red pixel, fill blue:
                const float diff1  = fabsf(ntl.z - nbr.z) + fabsf(ntl.y - color.y) + fabsf(nbr.y - color.y);
                const float guess1 = ntl.z + nbr.z + 2.0f*color.y - ntl.y - nbr.y;
                const float diff2  = fabsf(ntr.z - nbl.z) + fabsf(ntr.y - color.y) + fabsf(nbl.y - color.y);
                const float guess2 = ntr.z + nbl.z + 2.0f*color.y - ntr.y - nbl.y;
                if     (diff1 > diff2) color.z = guess2 * 0.5f;
                else if(diff1 < diff2) color.z = guess1 * 0.5f;
                else color.z = (guess1 + guess2)*0.25f;
            }
            else // c == 2, blue pixel, fill red:
            {
                const float diff1  = fabsf(ntl.x - nbr.x) + fabsf(ntl.y - color.y) + fabsf(nbr.y - color.y);
                const float guess1 = ntl.x + nbr.x + 2.0f*color.y - ntl.y - nbr.y;
                const float diff2  = fabsf(ntr.x - nbl.x) + fabsf(ntr.y - color.y) + fabsf(nbl.y - color.y);
                const float guess2 = ntr.x + nbl.x + 2.0f*color.y - ntr.y - nbl.y;
                if     (diff1 > diff2) color.x = guess2 * 0.5f;
                else if(diff1 < diff2) color.x = guess1 * 0.5f;
                else color.x = (guess1 + guess2)*0.25f;
            }
        }
    }
    output[y * width + x] = make_float4(fmaxf(color.x, 0.0f), fmaxf(color.y, 0.0f), fmaxf(color.z, 0.0f), 1.0f);
}

// Additional utility kernels for blending and masking
__global__ void write_blended_dual_kernel(float4* high, float4* low, float4* output,
                                         int width, int height, float* mask, int showmask)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if((col >= width) || (row >= height)) return;
    const int idx = row * width + col;

    const float4 high_val = high[idx];
    const float4 low_val = low[idx];
    const float blender = mask[idx];
    float4 data = make_float4(
        (1.0f - blender) * low_val.x + blender * high_val.x,
        (1.0f - blender) * low_val.y + blender * high_val.y,
        (1.0f - blender) * low_val.z + blender * high_val.z,
        showmask ? mask[idx] : 0.0f
    );

    output[idx] = make_float4(fmaxf(data.x, 0.0f), fmaxf(data.y, 0.0f), fmaxf(data.z, 0.0f), data.w);
}

__global__ void calc_Y0_mask_kernel(float* mask, float4* input, int width, int height, 
                                   float red, float green, float blue)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if((col >= width) || (row >= height)) return;
    const int idx = row * width + col;

    const float4 pt = input[idx];
    const float val = fmaxf(pt.x / red, 0.0f)
                    + fmaxf(pt.y / green, 0.0f)
                    + fmaxf(pt.z / blue, 0.0f);
    mask[idx] = dtcl_sqrt(val / 3.0f);
}

__global__ void calc_scharr_mask_kernel(float* input, float* output, int width, int height)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if((col >= width) || (row >= height)) return;

    const int oidx = row * width + col;
    const int incol = max(1, min(col, width - 2));
    const int inrow = max(1, min(row, height - 2));
    const int idx = inrow * width + incol;
    const float gx = 47.0f / 255.0f * (input[idx-width-1] - input[idx-width+1] + input[idx+width-1] - input[idx+width+1])
                  + 162.0f / 255.0f * (input[idx-1] - input[idx+1]);
    const float gy = 47.0f / 255.0f * (input[idx-width-1] - input[idx+width-1] + input[idx-width+1] - input[idx+width+1])
                  + 162.0f / 255.0f * (input[idx-width] - input[idx+width]);
    const float gradient_magnitude = dt_fast_hypot(gx, gy);
    output[oidx] = clipf(gradient_magnitude / 16.0f);
}

__global__ void calc_detail_blend_kernel(float* input, float* output, int width, int height, 
                                        float threshold, int detail)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if((col >= width) || (row >= height)) return;

    const int idx = row * width + col;

    const float blend = clipf(calcBlendFactor(input[idx], threshold));
    output[idx] = detail ? blend : 1.0f - blend;
}

// Complete RCD demosaic function implementation - following darktable flow
torch::Tensor rcd_demosaic_cuda(torch::Tensor input, uint32_t filters, float input_scale, float output_scale) {
    // Input validation
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (H, W, 1)");
    TORCH_CHECK(input.size(2) == 1, "Input must have single channel (raw Bayer)");
    
    // Ensure input is contiguous
    input = input.contiguous();
    
    const int height = input.size(0);
    const int width = input.size(1);
    const int RCD_MARGIN = 7;
    
    // Create output tensor (H, W, 4) for RGBA
    auto output = torch::zeros({height, width, 4}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // Create all temporary buffers needed for RCD algorithm
    const auto buffer_opts = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto dev_tmp = torch::zeros({height, width, 4}, buffer_opts);
    auto cfa = torch::zeros({height, width}, buffer_opts);
    auto rgb0 = torch::zeros({height, width}, buffer_opts);
    auto rgb1 = torch::zeros({height, width}, buffer_opts);
    auto rgb2 = torch::zeros({height, width}, buffer_opts);
    auto VH_dir = torch::zeros({height, width}, buffer_opts);
    auto PQ_dir = torch::zeros({height, width}, buffer_opts);
    auto VP_diff = torch::zeros({height, width}, buffer_opts);
    auto HQ_diff = torch::zeros({height, width}, buffer_opts);
    
    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Setup grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    dim3 grid_half((width/2 + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Step 1: Border interpolation - initial RGB estimation for borders
    border_interpolate_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), reinterpret_cast<float4*>(dev_tmp.data_ptr<float>()), 
        width, height, filters, 3);
    
    // Step 2: RCD border green interpolation with shared memory
    const int green_stride = block.x + 2*3;
    const int green_shared_size = green_stride * (block.y + 2*3) * sizeof(float);
    
    rcd_border_green_kernel<<<grid, block, green_shared_size, stream>>>(
        input.data_ptr<float>(), reinterpret_cast<float4*>(dev_tmp.data_ptr<float>()), 
        width, height, filters, 32);
    
    // Step 3: RCD border red/blue interpolation  
    const int redblue_stride = block.x + 2;
    const int redblue_shared_size = redblue_stride * (block.y + 2) * sizeof(float4);
    
    rcd_border_redblue_kernel<<<grid, block, redblue_shared_size, stream>>>(
        reinterpret_cast<float4*>(dev_tmp.data_ptr<float>()), 
        reinterpret_cast<float4*>(output.data_ptr<float>()), 
        width, height, filters, 16);
    
    // Step 4: Populate CFA and RGB data with scaling
    rcd_populate_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), cfa.data_ptr<float>(), rgb0.data_ptr<float>(), 
        rgb1.data_ptr<float>(), rgb2.data_ptr<float>(), width, height, filters, input_scale);
    
    // Step 5: RCD Algorithm Step 1.1 - Calculate squared vertical and horizontal high pass filters
    rcd_step_1_1_kernel<<<grid, block, 0, stream>>>(
        cfa.data_ptr<float>(), VP_diff.data_ptr<float>(), HQ_diff.data_ptr<float>(), width, height);
    
    // Step 6: RCD Algorithm Step 1.2 - Calculate vertical and horizontal local discrimination  
    rcd_step_1_2_kernel<<<grid, block, 0, stream>>>(
        VH_dir.data_ptr<float>(), VP_diff.data_ptr<float>(), HQ_diff.data_ptr<float>(), width, height);
    
    // Step 7: RCD Algorithm Step 2.1 - Low pass filter (note: width/2 grid)
    rcd_step_2_1_kernel<<<grid_half, block, 0, stream>>>(
        PQ_dir.data_ptr<float>(), cfa.data_ptr<float>(), width, height, filters);
    
    // Step 8: RCD Algorithm Step 3.1 - Populate green channel at blue and red CFA positions
    rcd_step_3_1_kernel<<<grid_half, block, 0, stream>>>(
        PQ_dir.data_ptr<float>(), cfa.data_ptr<float>(), rgb1.data_ptr<float>(), 
        VH_dir.data_ptr<float>(), width, height, filters);
    
    // Step 9: RCD Algorithm Step 4.1 - Calculate squared P/Q diagonals high pass filter
    rcd_step_4_1_kernel<<<grid_half, block, 0, stream>>>(
        cfa.data_ptr<float>(), VP_diff.data_ptr<float>(), HQ_diff.data_ptr<float>(), 
        width, height, filters);
    
    // Step 10: RCD Algorithm Step 4.2 - Calculate P/Q diagonal local discrimination
    rcd_step_4_2_kernel<<<grid_half, block, 0, stream>>>(
        PQ_dir.data_ptr<float>(), VP_diff.data_ptr<float>(), HQ_diff.data_ptr<float>(), 
        width, height, filters);
    
    // Step 11: RCD Algorithm Step 5.1 - Populate red and blue channels at blue and red CFA positions
    rcd_step_5_1_kernel<<<grid_half, block, 0, stream>>>(
        PQ_dir.data_ptr<float>(), rgb0.data_ptr<float>(), rgb1.data_ptr<float>(), 
        rgb2.data_ptr<float>(), width, height, filters);
    
    // Step 12: RCD Algorithm Step 5.2 - Populate red and blue channels at green CFA positions  
    rcd_step_5_2_kernel<<<grid_half, block, 0, stream>>>(
        VH_dir.data_ptr<float>(), rgb0.data_ptr<float>(), rgb1.data_ptr<float>(), 
        rgb2.data_ptr<float>(), width, height, filters);
    
    // Step 13: Write final output with scaling (only inner region, preserving borders)
    rcd_write_output_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<float4*>(output.data_ptr<float>()), rgb0.data_ptr<float>(), 
        rgb1.data_ptr<float>(), rgb2.data_ptr<float>(), width, height, output_scale, RCD_MARGIN);
    
    // Synchronize to ensure completion
    cudaStreamSynchronize(stream);
    
    return output;
}
