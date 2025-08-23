#include <cuda_runtime.h>
#include "color_conversions.h"

// RGB to XYZ conversion (sRGB D65)
__device__ float3 rgb_to_xyz(float3 rgb) {
    // sRGB to linear RGB
    float3 linear_rgb;
    linear_rgb.x = (rgb.x > 0.04045f) ? powf((rgb.x + 0.055f) / 1.055f, 2.4f) : rgb.x / 12.92f;
    linear_rgb.y = (rgb.y > 0.04045f) ? powf((rgb.y + 0.055f) / 1.055f, 2.4f) : rgb.y / 12.92f;
    linear_rgb.z = (rgb.z > 0.04045f) ? powf((rgb.z + 0.055f) / 1.055f, 2.4f) : rgb.z / 12.92f;
    
    // RGB to XYZ (sRGB D65)
    float3 xyz;
    xyz.x = 0.4124564f * linear_rgb.x + 0.3575761f * linear_rgb.y + 0.1804375f * linear_rgb.z;
    xyz.y = 0.2126729f * linear_rgb.x + 0.7151522f * linear_rgb.y + 0.0721750f * linear_rgb.z;
    xyz.z = 0.0193339f * linear_rgb.x + 0.1191920f * linear_rgb.y + 0.9503041f * linear_rgb.z;
    
    return xyz;
}

// XYZ to LAB conversion  
__device__ float3 xyz_to_lab(float3 xyz) {
    // Normalize by D65 illuminant
    xyz.x /= 0.95047f;
    xyz.y /= 1.0f;
    xyz.z /= 1.08883f;
    
    // XYZ to LAB
    float fx = (xyz.x > 0.008856f) ? powf(xyz.x, 1.0f/3.0f) : (7.787f * xyz.x + 16.0f/116.0f);
    float fy = (xyz.y > 0.008856f) ? powf(xyz.y, 1.0f/3.0f) : (7.787f * xyz.y + 16.0f/116.0f);
    float fz = (xyz.z > 0.008856f) ? powf(xyz.z, 1.0f/3.0f) : (7.787f * xyz.z + 16.0f/116.0f);
    
    float3 lab;
    lab.x = 116.0f * fy - 16.0f;
    lab.y = 500.0f * (fx - fy);
    lab.z = 200.0f * (fy - fz);
    
    return lab;
}

// LAB to XYZ conversion
__device__ float3 lab_to_xyz(float3 lab) {
    float fy = (lab.x + 16.0f) / 116.0f;
    float fx = lab.y / 500.0f + fy;
    float fz = fy - lab.z / 200.0f;
    
    float3 xyz;
    xyz.x = (fx*fx*fx > 0.008856f) ? fx*fx*fx : (fx - 16.0f/116.0f) / 7.787f;
    xyz.y = (fy*fy*fy > 0.008856f) ? fy*fy*fy : (fy - 16.0f/116.0f) / 7.787f;
    xyz.z = (fz*fz*fz > 0.008856f) ? fz*fz*fz : (fz - 16.0f/116.0f) / 7.787f;
    
    // Multiply by D65 illuminant
    xyz.x *= 0.95047f;
    xyz.y *= 1.0f;
    xyz.z *= 1.08883f;
    
    return xyz;
}

// XYZ to RGB conversion (sRGB D65)
__device__ float3 xyz_to_rgb(float3 xyz) {
    // XYZ to linear RGB (sRGB D65)
    float3 linear_rgb;
    linear_rgb.x =  3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z;
    linear_rgb.y = -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z;
    linear_rgb.z =  0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z;
    
    // Linear RGB to sRGB
    float3 rgb;
    rgb.x = (linear_rgb.x > 0.0031308f) ? 1.055f * powf(linear_rgb.x, 1.0f/2.4f) - 0.055f : 12.92f * linear_rgb.x;
    rgb.y = (linear_rgb.y > 0.0031308f) ? 1.055f * powf(linear_rgb.y, 1.0f/2.4f) - 0.055f : 12.92f * linear_rgb.y;
    rgb.z = (linear_rgb.z > 0.0031308f) ? 1.055f * powf(linear_rgb.z, 1.0f/2.4f) - 0.055f : 12.92f * linear_rgb.z;
    
    // Clamp to [0,1]
    rgb.x = fmaxf(0.0f, fminf(1.0f, rgb.x));
    rgb.y = fmaxf(0.0f, fminf(1.0f, rgb.y));
    rgb.z = fmaxf(0.0f, fminf(1.0f, rgb.z));
    
    return rgb;
}

// Direct RGB to LAB conversion (combines RGB→XYZ→LAB)
__device__ float3 rgb_to_lab(float3 rgb) {
    // Convert RGB to XYZ first
    float3 xyz = rgb_to_xyz(rgb);

    // Convert XYZ to LAB
    return xyz_to_lab(xyz);
}

// Direct LAB to RGB conversion (combines LAB→XYZ→RGB)
__device__ float3 lab_to_rgb(float3 lab) {
    // Convert LAB to XYZ first
    float3 xyz = lab_to_xyz(lab);

    // Convert XYZ to RGB
    return xyz_to_rgb(xyz);
}

// Direct LAB L component from RGB (most efficient for luminance-only use)
__device__ float rgb_to_lab_l(float3 rgb) {
    // Convert RGB to linear RGB
    float3 linear_rgb;
    linear_rgb.x = (rgb.x > 0.04045f) ? powf((rgb.x + 0.055f) / 1.055f, 2.4f) : rgb.x / 12.92f;
    linear_rgb.y = (rgb.y > 0.04045f) ? powf((rgb.y + 0.055f) / 1.055f, 2.4f) : rgb.y / 12.92f;
    linear_rgb.z = (rgb.z > 0.04045f) ? powf((rgb.z + 0.055f) / 1.055f, 2.4f) : rgb.z / 12.92f;

    // Compute XYZ Y component (luminance)
    float y_xyz = 0.2126729f * linear_rgb.x + 0.7151522f * linear_rgb.y + 0.0721750f * linear_rgb.z;

    // Normalize by D65 reference white and apply LAB L transformation
    float y_norm = y_xyz / 1.0f;  // D65 Y = 1.0
    float fy = (y_norm > 0.008856f) ? powf(y_norm, 1.0f/3.0f) : (7.787f * y_norm + 16.0f/116.0f);
    float l_lab = 116.0f * fy - 16.0f;

    // Return L in 0-1 range
    return l_lab / 100.0f;
}

// Generic template kernel for all color space conversions
template <typename Converter>
__global__ void color_conversion_kernel(
    float* input,     // Input (H*W*3)
    float* output,    // Output (H*W*3)
    const int width,
    const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    const int idx = (y * width + x) * 3;
    float3 input_px = make_float3(input[idx], input[idx + 1], input[idx + 2]);

    Converter converter;
    float3 output_px = converter(input_px);

    output[idx] = output_px.x;
    output[idx + 1] = output_px.y;
    output[idx + 2] = output_px.z;
}

// Templated PyTorch wrapper for all color conversions
template <typename Converter>
torch::Tensor convert_color_space(const torch::Tensor& input, const char* kernel_name) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 3, "Input must be (H, W, 3)");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const int height = input.size(0);
    const int width = input.size(1);

    auto output = torch::empty_like(input);

    constexpr int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);

    color_conversion_kernel<Converter><<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + kernel_name + ": " + cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA synchronization error in ") + kernel_name + ": " + cudaGetErrorString(err));
    }

    return output;
}





// Extract luminance from RGB image
__global__ void compute_luminance_kernel(
    float* rgb,       // RGB (H*W*3)
    float* luminance, // L (H*W*1)
    const int width,
    const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    // Read RGB pixel (PyTorch column-major layout: [height, width, channels])
    const int rgb_idx = (y * width + x) * 3;
    float3 rgb_px = make_float3(rgb[rgb_idx], rgb[rgb_idx + 1], rgb[rgb_idx + 2]);
    
    // Clamp RGB values to valid range to prevent NaN/inf
    rgb_px.x = fmaxf(0.0f, fminf(1.0f, rgb_px.x));
    rgb_px.y = fmaxf(0.0f, fminf(1.0f, rgb_px.y));
    rgb_px.z = fmaxf(0.0f, fminf(1.0f, rgb_px.z));
    
    // Compute LAB L component using dedicated function
    float lum = rgb_to_lab_l(rgb_px);

    // Store luminance (PyTorch column-major layout: [height, width])
    luminance[y * width + x] = lum;
}

// Update RGB image with modified luminance
__global__ void modify_luminance_kernel(
    float* rgb,           // Original RGB (H*W*3)
    float* new_luminance, // New L channel (H*W*1)
    float* result,        // Updated RGB (H*W*3)
    const int width,
    const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    const int idx = y * width + x;
    const int rgb_idx = (y * width + x) * 3;

    // Read original RGB and convert to LAB directly to get a,b
    float3 rgb_orig = make_float3(rgb[rgb_idx], rgb[rgb_idx + 1], rgb[rgb_idx + 2]);
    float3 lab_orig = rgb_to_lab(rgb_orig);
    
    // Create new LAB with modified L, original a,b
    float new_l = new_luminance[idx];
    // Clamp luminance to valid range
    new_l = fmaxf(0.0f, fminf(1.0f, new_l));
    
    float3 lab_new = make_float3(
        new_l * 100.0f,  // Convert back to 0-100 range
        lab_orig.y,      // Keep original a
        lab_orig.z       // Keep original b
    );
    
    // Convert back to RGB directly
    float3 rgb_new = lab_to_rgb(lab_new);
    
    // Clamp output RGB values to valid range
    rgb_new.x = fmaxf(0.0f, fminf(1.0f, rgb_new.x));
    rgb_new.y = fmaxf(0.0f, fminf(1.0f, rgb_new.y));
    rgb_new.z = fmaxf(0.0f, fminf(1.0f, rgb_new.z));
    
    // Store result
    result[rgb_idx] = rgb_new.x;
    result[rgb_idx + 1] = rgb_new.y;
    result[rgb_idx + 2] = rgb_new.z;
}

// PyTorch tensor wrapper functions
torch::Tensor compute_luminance(const torch::Tensor& rgb) {
    TORCH_CHECK(rgb.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(rgb.dim() == 3 && rgb.size(2) == 3, "Input must be (H, W, 3)");
    TORCH_CHECK(rgb.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(rgb.is_contiguous(), "Input tensor must be contiguous");
    
    const int height = rgb.size(0);
    const int width = rgb.size(1);
    
    auto luminance = torch::zeros({height, width}, rgb.options());
    
    constexpr int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);
    
    compute_luminance_kernel<<<grid, block>>>(
        rgb.data_ptr<float>(), luminance.data_ptr<float>(), width, height);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in compute_luminance_kernel: ") + cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA synchronization error in compute_luminance_kernel: ") + cudaGetErrorString(err));
    }
    
    return luminance;
}

torch::Tensor modify_luminance(const torch::Tensor& rgb, const torch::Tensor& new_luminance) {
    TORCH_CHECK(rgb.dtype() == torch::kFloat32, "RGB input must be float32");
    TORCH_CHECK(new_luminance.dtype() == torch::kFloat32, "Luminance input must be float32");
    TORCH_CHECK(rgb.dim() == 3 && rgb.size(2) == 3, "RGB input must be (H, W, 3)");
    TORCH_CHECK(new_luminance.dim() == 2, "Luminance input must be (H, W)");
    TORCH_CHECK(rgb.is_cuda() && new_luminance.is_cuda(), "Inputs must be on CUDA device");
    TORCH_CHECK(rgb.is_contiguous() && new_luminance.is_contiguous(), "Input tensors must be contiguous");
    
    const int height = rgb.size(0);
    const int width = rgb.size(1);


   cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error BEFORE modify_luminance_kernel: ") + cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA synchronization error in modify_luminance_kernel: ") + cudaGetErrorString(err));
    }

    TORCH_CHECK(new_luminance.size(0) == height && new_luminance.size(1) == width, 
                "Luminance dimensions must match RGB");
    
    // Validate tensor properties
    TORCH_CHECK(rgb.numel() == height * width * 3, "RGB tensor size mismatch");
    TORCH_CHECK(new_luminance.numel() == height * width, "Luminance tensor size mismatch");
    TORCH_CHECK(rgb.data_ptr<float>() != nullptr, "RGB tensor data pointer is null");
    TORCH_CHECK(new_luminance.data_ptr<float>() != nullptr, "Luminance tensor data pointer is null");
    
    auto result = torch::zeros_like(rgb);
    
    constexpr int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);
    
    modify_luminance_kernel<<<grid, block>>>(
        rgb.data_ptr<float>(), new_luminance.data_ptr<float>(), 
        result.data_ptr<float>(), width, height);
    
    // // Check for CUDA errors
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     throw std::runtime_error(std::string("CUDA error in modify_luminance_kernel: ") + cudaGetErrorString(err));
    // }
    
    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA synchronization error in modify_luminance_kernel: ") + cudaGetErrorString(err));
    }
    
    return result;
}





struct ConvertRgbToXyz {
    __device__ float3 operator()(float3 input) const {
        return rgb_to_xyz(input);
    }
};

struct ConvertXyzToLab {
    __device__ float3 operator()(float3 input) const {
        return xyz_to_lab(input);
    }
};

struct ConvertLabToXyz {
    __device__ float3 operator()(float3 input) const {
        return lab_to_xyz(input);
    }
};

struct ConvertXyzToRgb {
    __device__ float3 operator()(float3 input) const {
        return xyz_to_rgb(input);
    }
};

struct ConvertRgbToLab {
    __device__ float3 operator()(float3 input) const {
        return rgb_to_lab(input);
    }
};

struct ConvertLabToRgb {
    __device__ float3 operator()(float3 input) const {
        return lab_to_rgb(input);
    }
};

// PyTorch tensor wrapper for RGB to XYZ
torch::Tensor rgb_to_xyz(const torch::Tensor& rgb) {
    return convert_color_space<ConvertRgbToXyz>(rgb, "rgb_to_xyz");
}

// PyTorch tensor wrapper for XYZ to LAB
torch::Tensor xyz_to_lab(const torch::Tensor& xyz) {
    return convert_color_space<ConvertXyzToLab>(xyz, "xyz_to_lab");
}

// PyTorch tensor wrapper for LAB to XYZ
torch::Tensor lab_to_xyz(const torch::Tensor& lab) {
    return convert_color_space<ConvertLabToXyz>(lab, "lab_to_xyz");
}

// PyTorch tensor wrapper for XYZ to RGB
torch::Tensor xyz_to_rgb(const torch::Tensor& xyz) {
    return convert_color_space<ConvertXyzToRgb>(xyz, "xyz_to_rgb");
}

// PyTorch tensor wrapper for RGB to LAB
torch::Tensor rgb_to_lab(const torch::Tensor& rgb) {
    return convert_color_space<ConvertRgbToLab>(rgb, "rgb_to_lab");
}

// PyTorch tensor wrapper for LAB to RGB
torch::Tensor lab_to_rgb(const torch::Tensor& lab) {
    return convert_color_space<ConvertLabToRgb>(lab, "lab_to_rgb");
}




