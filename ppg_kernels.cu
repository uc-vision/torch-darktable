/*
 * PPG Demosaic CUDA Kernels
 * Converted from OpenCL kernels for better PyTorch integration
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// FC macro for Bayer pattern (from darktable)
#define FC(row, col, filters) ((filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1)) & 3)

/**
 * Pre-median filtering kernel
 */
__global__ void pre_median_kernel(
    float* input,
    float* output, 
    int width,
    int height,
    uint32_t filters,
    float threshold
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = y * width + x;
    
    if (threshold <= 0.0f) {
        output[idx] = input[idx];
        return;
    }
    
    // Simple median filter implementation
    float values[9];
    int count = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                values[count++] = input[ny * width + nx];
            }
        }
    }
    
    // Simple bubble sort for small array
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                float temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    
    float median = values[count / 2];
    float current = input[idx];
    
    output[idx] = (fabsf(current - median) > threshold) ? median : current;
}

/**
 * PPG Green channel interpolation kernel
 */
__global__ void ppg_demosaic_green_kernel(
    float* input,
    float* output,
    int width,
    int height,
    uint32_t filters
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = y * width + x;
    const int f = FC(y, x, filters);
    
    if (f == 1) { // Green pixel
        output[idx] = input[idx];
        return;
    }
    
    // Interpolate green at red/blue pixels
    float sum = 0.0f;
    int count = 0;
    
    // Check 4-connected neighbors for green values
    const int dx[] = {0, 1, 0, -1};
    const int dy[] = {-1, 0, 1, 0};
    
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            if (FC(ny, nx, filters) == 1) { // Green neighbor
                sum += input[ny * width + nx];
                count++;
            }
        }
    }
    
    output[idx] = (count > 0) ? sum / count : input[idx];
}

/**
 * PPG Red/Blue channel interpolation kernel
 */
__global__ void ppg_demosaic_redblue_kernel(
    float* green_input,  // Already has green interpolated
    float* raw_input,    // Original raw data
    float4* output,      // RGBA output
    int width,
    int height,
    uint32_t filters
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = y * width + x;
    const int f = FC(y, x, filters);
    
    float4 pixel = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Green channel (already interpolated)
    pixel.y = green_input[idx];
    
    if (f == 0) { // Red pixel
        pixel.x = raw_input[idx];
        
        // Interpolate blue
        float sum = 0.0f;
        int count = 0;
        
        // Check diagonal neighbors for blue
        const int dx[] = {-1, 1, -1, 1};
        const int dy[] = {-1, -1, 1, 1};
        
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (FC(ny, nx, filters) == 2) { // Blue neighbor
                    sum += raw_input[ny * width + nx];
                    count++;
                }
            }
        }
        
        pixel.z = (count > 0) ? sum / count : 0.0f;
        
    } else if (f == 2) { // Blue pixel
        pixel.z = raw_input[idx];
        
        // Interpolate red
        float sum = 0.0f;
        int count = 0;
        
        // Check diagonal neighbors for red
        const int dx[] = {-1, 1, -1, 1};
        const int dy[] = {-1, -1, 1, 1};
        
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (FC(ny, nx, filters) == 0) { // Red neighbor
                    sum += raw_input[ny * width + nx];
                    count++;
                }
            }
        }
        
        pixel.x = (count > 0) ? sum / count : 0.0f;
        
    } else { // Green pixel
        // Interpolate both red and blue
        float red_sum = 0.0f, blue_sum = 0.0f;
        int red_count = 0, blue_count = 0;
        
        // Check 4-connected neighbors
        const int dx[] = {0, 1, 0, -1};
        const int dy[] = {-1, 0, 1, 0};
        
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nf = FC(ny, nx, filters);
                if (nf == 0) { // Red
                    red_sum += raw_input[ny * width + nx];
                    red_count++;
                } else if (nf == 2) { // Blue
                    blue_sum += raw_input[ny * width + nx];
                    blue_count++;
                }
            }
        }
        
        pixel.x = (red_count > 0) ? red_sum / red_count : 0.0f;
        pixel.z = (blue_count > 0) ? blue_sum / blue_count : 0.0f;
    }
    
    output[idx] = pixel;
}

/**
 * Border interpolation kernel for edge pixels
 */
__global__ void border_interpolate_kernel(
    float4* data,
    int width,
    int height,
    uint32_t filters,
    int border
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Only process border pixels
    if (x >= border && x < width - border && y >= border && y < height - border) {
        return;
    }
    
    const int idx = y * width + x;
    
    // Simple border handling: copy from nearest valid pixel
    int src_x = max(border, min(width - border - 1, x));
    int src_y = max(border, min(height - border - 1, y));
    int src_idx = src_y * width + src_x;
    
    data[idx] = data[src_idx];
}

// Host wrapper functions
extern "C" {

void launch_pre_median(
    float* input,
    float* output,
    int width,
    int height,
    uint32_t filters,
    float threshold,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    pre_median_kernel<<<grid, block, 0, stream>>>(
        input, output, width, height, filters, threshold
    );
}

void launch_ppg_green(
    float* input,
    float* output,
    int width,
    int height,
    uint32_t filters,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    ppg_demosaic_green_kernel<<<grid, block, 0, stream>>>(
        input, output, width, height, filters
    );
}

void launch_ppg_redblue(
    float* green_input,
    float* raw_input,
    float4* output,
    int width,
    int height,
    uint32_t filters,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    ppg_demosaic_redblue_kernel<<<grid, block, 0, stream>>>(
        green_input, raw_input, output, width, height, filters
    );
}

void launch_border_interpolate(
    float4* data,
    int width,
    int height,
    uint32_t filters,
    int border,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    border_interpolate_kernel<<<grid, block, 0, stream>>>(
        data, width, height, filters, border
    );
}

} // extern "C"
