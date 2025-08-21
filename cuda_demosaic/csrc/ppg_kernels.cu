/*
 * PPG Demosaic CUDA Kernels
 * Converted from OpenCL kernels for better PyTorch integration
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

// FC macro for Bayer pattern (from darktable)
#define FC(row, col, filters) ((filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1)) & 3)

__constant__ int glim[5] = { 0, 1, 2, 1, 0 };

#define SWAP(a, b)                \
  {                               \
    const float tmp = (b);        \
    (b) = (a);                    \
    (a) = tmp;                    \
  }

/**
 * Pre-median filtering kernel - exact translation from darktable OpenCL
 */
__global__ void pre_median_kernel(
    float* input,
    float* output, 
    int width,
    int height,
    uint32_t filters,
    float threshold
) {
    extern __shared__ float median_buffer[];
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
    for(int n = 0; n <= maxbuf/lsz; n++)
    {
        const int bufidx = n * lsz + l;
        if(bufidx >= maxbuf) continue;
        const int xx = xul + bufidx % stride;
        const int yy = yul + bufidx / stride;
        median_buffer[bufidx] = (xx >= 0 && yy >= 0 && xx < width && yy < height) ? input[yy * width + xx] : 0.0f;
    }

    // center buffer around current x,y-Pixel
    float* centered_median = median_buffer + (ylid + 2) * stride + xlid + 2;

    __syncthreads();

    if(x >= width || y >= height) return;

    const int *lim = glim;

    const int c = FC(y, x, filters);

    float med[9];

    int cnt = 0;

    for(int k = 0, i = 0; i < 5; i++)
    {
        for(int j = -lim[i]; j <= lim[i]; j += 2)
        {
            if(fabsf(centered_median[stride * (i - 2) + j] - centered_median[0]) < threshold)
            {
                med[k++] = centered_median[stride * (i - 2) + j];
                cnt++;
            }
            else
                med[k++] = 64.0f + centered_median[stride * (i - 2) + j];
        }
    }

    for(int i = 0; i < 8; i++)
        for(int ii = i + 1; ii < 9; ii++)
            if(med[i] > med[ii]) SWAP(med[i], med[ii]);

    float color = (c & 1) ? (cnt == 1 ? med[4] - 64.0f : med[(cnt - 1) / 2]) : centered_median[0];

    output[y * width + x] = fmaxf(color, 0.0f);
}

#undef SWAP

/**
 * fill greens pass of pattern pixel grouping - exact translation from darktable OpenCL
 * in (float) or (float4).x -> out (float4)
 */
__global__ void ppg_demosaic_green_kernel(
    float* input,
    float4* output,
    int width,
    int height,
    uint32_t filters
) {
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
        green_buffer[bufidx] = (xx >= 0 && yy >= 0 && xx < width && yy < height) ? input[yy * width + xx] : 0.0f;
    }

    // center buffer around current x,y-Pixel
    float* centered_green = green_buffer + (ylid + 3) * stride + xlid + 3;

    __syncthreads();

    // make sure we dont write the outermost 3 pixels
    if(x >= width - 3 || x < 3 || y >= height - 3 || y < 3) return;
    
    // process all non-green pixels
    const int row = y;
    const int col = x;
    const int c = FC(row, col, filters);
    float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // output color

    const float pc = centered_green[0];

    if     (c == 0) color.x = pc; // red
    else if(c == 1) color.y = pc; // green1
    else if(c == 2) color.z = pc; // blue
    else            color.y = pc; // green2

    // fill green layer for red and blue pixels:
    if(c == 0 || c == 2)
    {
        // look up horizontal and vertical neighbours, sharpened weight:
        const float pym  = centered_green[-1 * stride];
        const float pym2 = centered_green[-2 * stride];
        const float pym3 = centered_green[-3 * stride];
        const float pyM  = centered_green[ 1 * stride];
        const float pyM2 = centered_green[ 2 * stride];
        const float pyM3 = centered_green[ 3 * stride];
        const float pxm  = centered_green[-1];
        const float pxm2 = centered_green[-2];
        const float pxm3 = centered_green[-3];
        const float pxM  = centered_green[ 1];
        const float pxM2 = centered_green[ 2];
        const float pxM3 = centered_green[ 3];
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

/**
 * fills the reds and blues in the gaps (done after ppg_demosaic_green) - exact translation from darktable OpenCL
 * in (float4) -> out (float4)
 */
__global__ void ppg_demosaic_redblue_kernel(
    float4* input,
    float4* output,
    int width,
    int height,
    uint32_t filters
) {
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
        redblue_buffer[bufidx] = (xx >= 0 && yy >= 0 && xx < width && yy < height) ? input[yy * width + xx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // center buffer around current x,y-Pixel
    float4* centered_redblue = redblue_buffer + (ylid + 1) * stride + xlid + 1;

    __syncthreads();

    if(x >= width || y >= height) return;
    const int row = y;
    const int col = x;
    const int c = FC(row, col, filters);
    float4 color = centered_redblue[0];
    if(x == 0 || y == 0 || x == (width-1) || y == (height-1))
    {
        output[y * width + x] = make_float4(fmaxf(color.x, 0.0f), fmaxf(color.y, 0.0f), fmaxf(color.z, 0.0f), 1.0f);
        return;
    }

    if(c == 1 || c == 3)
    { // calculate red and blue for green pixels:
        // need 4-nbhood:
        const float4 nt = centered_redblue[-stride];
        const float4 nb = centered_redblue[ stride];
        const float4 nl = centered_redblue[-1];
        const float4 nr = centered_redblue[ 1];
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
        const float4 ntl = centered_redblue[-stride - 1];
        const float4 ntr = centered_redblue[-stride + 1];
        const float4 nbl = centered_redblue[ stride - 1];
        const float4 nbr = centered_redblue[ stride + 1];

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
    output[y * width + x] = make_float4(fmaxf(color.x, 0.0f), fmaxf(color.y, 0.0f), fmaxf(color.z, 0.0f), 1.0f);
}

/**
 * Demosaic image border - exact translation from darktable OpenCL
 */
__global__ void border_interpolate_kernel(
    float* input,
    float4* output,
    int width,
    int height,
    uint32_t filters,
    int border
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    const int avgwindow = 1;

    if(x >= border && x < width-border && y >= border && y < height-border) return;

    float4 o;
    float sum[4] = { 0.0f };
    int count[4] = { 0 };

    for (int j=y-avgwindow; j<=y+avgwindow; j++) for (int i=x-avgwindow; i<=x+avgwindow; i++)
    {
        if (j>=0 && i>=0 && j<height && i<width)
        {
            const int f = FC(j,i,filters);
            sum[f] += fmaxf(0.0f, input[j * width + i]);
            count[f]++;
        }
    }

    const float i = fmaxf(0.0f, input[y * width + x]);
    o.x = count[0] > 0 ? sum[0]/count[0] : i;
    o.y = count[1]+count[3] > 0 ? (sum[1]+sum[3])/(count[1]+count[3]) : i;
    o.z = count[2] > 0 ? sum[2]/count[2] : i;

    const int f = FC(y,x,filters);

    if     (f == 0) o.x = i;
    else if(f == 1) o.y = i;
    else if(f == 2) o.z = i;
    else            o.y = i;

    output[y * width + x] = o;
}



// Complete demosaic function implementation - exact darktable flow
torch::Tensor ppg_demosaic_cuda(torch::Tensor input, uint32_t filters, float median_threshold) {
    // Input validation
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (H, W, 1)");
    TORCH_CHECK(input.size(2) == 1, "Input must have single channel (raw Bayer)");
    
    // Ensure input is contiguous
    input = input.contiguous();
    
    const int height = input.size(0);
    const int width = input.size(1);
    
    // Create output tensor (H, W, 4) for RGBA
    auto output = torch::zeros({height, width, 4}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // Create temporary buffer for median filtering
    auto temp_median = torch::zeros({height, width}, 
                                   torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Setup grid and block dimensions - must match darktable's local work group sizing
    dim3 block(16, 16);  // This needs to match OpenCL local size
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Calculate shared memory sizes for each kernel
    const int median_stride = block.x + 2*2;
    const int median_shared_size = median_stride * (block.y + 2*2) * sizeof(float);
    
    const int green_stride = block.x + 2*3;
    const int green_shared_size = green_stride * (block.y + 2*3) * sizeof(float);
    
    const int redblue_stride = block.x + 2;
    const int redblue_shared_size = redblue_stride * (block.y + 2) * sizeof(float4);
    
    auto* processing_input = input.data_ptr<float>();
    
    // Step 1: Border interpolation - demosaic border regions to RGB FIRST
    border_interpolate_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), reinterpret_cast<float4*>(output.data_ptr<float>()), 
        width, height, filters, 3);
    
    // Step 2: Pre-median filtering (optional) on raw data
    if (median_threshold > 0.0f) {
        pre_median_kernel<<<grid, block, median_shared_size, stream>>>(
            input.data_ptr<float>(), temp_median.data_ptr<float>(), 
            width, height, filters, median_threshold);
        processing_input = temp_median.data_ptr<float>();
    }
    
    // Step 3: Green channel interpolation - outputs to RGB buffer (overwrites borders)
    ppg_demosaic_green_kernel<<<grid, block, green_shared_size, stream>>>(
        processing_input, reinterpret_cast<float4*>(output.data_ptr<float>()), 
        width, height, filters);
    
    // Step 4: Red/Blue interpolation - final step using RGB buffer
    ppg_demosaic_redblue_kernel<<<grid, block, redblue_shared_size, stream>>>(
        reinterpret_cast<float4*>(output.data_ptr<float>()), 
        reinterpret_cast<float4*>(output.data_ptr<float>()), 
        width, height, filters);
    
    // Synchronize to ensure completion
    cudaStreamSynchronize(stream);
    
    return output;
}
