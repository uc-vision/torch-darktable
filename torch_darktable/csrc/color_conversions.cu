#include <cuda_runtime.h>
#include "color_conversions.h"
#include "cuda_utils.h"
#include "device_math.h"

// sRGB gamma correction (sRGB to linear)
__device__ __forceinline__ float srgb_to_linear(float x) {
    return (x > 0.04045f) ? powf((x + 0.055f) / 1.055f, 2.4f) : x / 12.92f;
}

// Vectorized sRGB gamma correction functions
__device__ __forceinline__ float3 srgb_to_linear(float3 rgb) {
    const float threshold = 0.04045f;
    const float a = 0.055f;
    const float gamma = 2.4f;
    const float linear_scale = 1.0f / 12.92f;
    
    // Vectorized comparison and operations
    bool3 use_pow = rgb > threshold;
    float3 pow_result = pow((rgb + a) / (1.0f + a), gamma);
    float3 linear_result = rgb * linear_scale;
    
    return select(use_pow, pow_result, linear_result);
}

// LAB f function for XYZ to LAB conversion
__device__ __forceinline__ float lab_f(float t) {
    return (t > 0.008856f) ? powf(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
}

// Efficient float3 overload for LAB f function
__device__ __forceinline__ float3 lab_f(float3 t) {
    const float threshold = 0.008856f;
    const float linear_mult = 7.787f;
    const float linear_add = 16.0f/116.0f;
    
    // Vectorized comparison and selection
    bool3 use_pow = t > threshold;
    float3 pow_result = pow(t, 1.0f/3.0f);
    float3 linear_result = t * linear_mult + linear_add;
    
    return select(use_pow, pow_result, linear_result);
}

// Inverse LAB f function for LAB to XYZ conversion
__device__ __forceinline__ float lab_f_inv(float t) {
    float t_cubed = t * t * t;
    return (t_cubed > 0.008856f) ? t_cubed : (t - 16.0f/116.0f) / 7.787f;
}

// Efficient float3 overload for LAB f inverse function
__device__ __forceinline__ float3 lab_f_inv(float3 t) {
    float3 t_cubed = t * t * t;  // Compute cubes once using float3 operators
    const float threshold = 0.008856f;
    const float linear_coeff = 16.0f/116.0f;
    const float linear_divisor = 7.787f;
    
    // Vectorized comparison and selection
    bool3 use_cubed = t_cubed > threshold;
    float3 linear_result = (t - linear_coeff) / linear_divisor;
    
    return select(use_cubed, t_cubed, linear_result);
}

// RGB to XYZ conversion (sRGB D65)
__device__ float3 rgb_to_xyz(float3 rgb) {
    // sRGB to linear RGB
    float3 linear_rgb = srgb_to_linear(rgb);
    
    // RGB to XYZ (sRGB D65) using matrix multiplication
    const float3x3 rgb_to_xyz_matrix(
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f
    );
    float3 xyz = rgb_to_xyz_matrix * linear_rgb;
    
    return xyz;
}

// XYZ to LAB conversion  
__device__ float3 xyz_to_lab(float3 xyz) {
    // Normalize by D65 illuminant
    const float3 d65_white = make_float3(0.95047f, 1.0f, 1.08883f);
    xyz = xyz / d65_white;
    
    // Apply LAB f function component-wise
    float3 f_xyz = lab_f(xyz);
    
    float3 lab = make_float3(
        116.0f * f_xyz.y - 16.0f,
        500.0f * (f_xyz.x - f_xyz.y),
        200.0f * (f_xyz.y - f_xyz.z)
    );
    
    return lab;
}

// LAB to XYZ conversion
__device__ float3 lab_to_xyz(float3 lab) {
    float fy = (lab.x + 16.0f) / 116.0f;
    float3 f_xyz = make_float3(
        lab.y / 500.0f + fy,
        fy,
        fy - lab.z / 200.0f
    );
    
    // Apply inverse LAB f function component-wise
    float3 xyz = lab_f_inv(f_xyz);
    
    // Multiply by D65 illuminant
    const float3 d65_white = make_float3(0.95047f, 1.0f, 1.08883f);
    return xyz * d65_white;
}

// Linear to sRGB gamma correction
__device__ __forceinline__ float linear_to_srgb(float x) {
    return (x > 0.0031308f) ? 1.055f * powf(x, 1.0f/2.4f) - 0.055f : 12.92f * x;
}


__device__ __forceinline__ float3 linear_to_srgb(float3 linear_rgb) {
    const float threshold = 0.0031308f;
    const float a = 0.055f;
    const float gamma = 1.0f / 2.4f;
    const float srgb_scale = 12.92f;
    
    // Vectorized comparison and operations
    bool3 use_pow = linear_rgb > threshold;
    float3 pow_result = (1.0f + a) * pow(linear_rgb, gamma) - a;
    float3 linear_result = linear_rgb * srgb_scale;
    
    return select(use_pow, pow_result, linear_result);
}

// XYZ to RGB conversion (sRGB D65)
__device__ float3 xyz_to_rgb(float3 xyz) {
    // XYZ to linear RGB (sRGB D65) using matrix multiplication
    const float3x3 xyz_to_rgb_matrix(
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f
    );
    float3 linear_rgb = xyz_to_rgb_matrix * xyz;
    
    // Linear RGB to sRGB
    float3 rgb = linear_to_srgb(linear_rgb);
    
    // Clamp to [0,1] using device_math utilities
    rgb = fmax(fmin(rgb, 1.0f), 0.0f);
    
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
    float3 linear_rgb = srgb_to_linear(rgb);

    // Compute XYZ Y component (luminance)
    float y_xyz = 0.2126729f * linear_rgb.x + 0.7151522f * linear_rgb.y + 0.0721750f * linear_rgb.z;

    // Normalize by D65 reference white and apply LAB L transformation
    float fy = lab_f(y_xyz);  // D65 Y = 1.0, so no division needed
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
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const int idx = (pos.y * width + pos.x) * 3;
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

    dim3 block = block_size_2d;
    dim3 grid = grid2d(width, height);

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
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    // Read RGB pixel (PyTorch column-major layout: [height, width, channels])
    const int rgb_idx = (pos.y * width + pos.x) * 3;
    float3 rgb_px = make_float3(rgb[rgb_idx], rgb[rgb_idx + 1], rgb[rgb_idx + 2]);
    
    // Clamp RGB values to valid range to prevent NaN/inf
    rgb_px = clamp01(rgb_px);
    
    // Compute LAB L component using dedicated function
    float lum = rgb_to_lab_l(rgb_px);

    // Store luminance (PyTorch column-major layout: [height, width])
    luminance[pos.y * width + pos.x] = lum;
}

// Update RGB image with modified luminance
__global__ void modify_luminance_kernel(
    float* rgb,           // Original RGB (H*W*3)
    float* new_luminance, // New L channel (H*W*1)
    float* result,        // Updated RGB (H*W*3)
    const int width,
    const int height)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const int idx = pos.y * width + pos.x;
    const int rgb_idx = (pos.y * width + pos.x) * 3;

    // Read original RGB and convert to LAB directly to get a,b
    float3 rgb_orig = make_float3(rgb[rgb_idx], rgb[rgb_idx + 1], rgb[rgb_idx + 2]);
    float3 lab_orig = rgb_to_lab(rgb_orig);
    
    // Create new LAB with modified L, original a,b
    float new_l = fmaxf(0.0f, fminf(1.0f, new_luminance[idx]));
    
    float3 lab_new = lab_orig;
    lab_new.x = new_l * 100.0f;  // Replace L component
    
    // Convert back to RGB directly
    float3 rgb_new = lab_to_rgb(lab_new);
    
    // Clamp output RGB values to valid range
    rgb_new = clamp01(rgb_new);
    
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
    
    dim3 block = block_size_2d;
    dim3 grid = grid2d(width, height);
    
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
    
    dim3 block = block_size_2d;
    dim3 grid = grid2d(width, height);
    
    modify_luminance_kernel<<<grid, block>>>(
        rgb.data_ptr<float>(), new_luminance.data_ptr<float>(), 
        result.data_ptr<float>(), width, height);
    
    
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

// Generic color transform with 3x3 matrix
__device__ float3 color_transform_3x3(float3 color, const float3x3& matrix_3x3) {
    float3 result = matrix_3x3 * color;
    // Clamp to [0,1] range
    return clamp01(result);
}

// Converter struct for generic 3x3 matrix transform
struct ConvertWithMatrix3x3 {
    float3x3 matrix_3x3;
    
    __device__ ConvertWithMatrix3x3(const float3x3& m) : matrix_3x3(m) {}
    
    __device__ float3 operator()(float3 input) const {
        return color_transform_3x3(input, matrix_3x3);
    }
};

// Specialized kernel for 3x3 matrix transform
__global__ void matrix_3x3_transform_kernel(
    float* input,     // Input (H*W*3)
    float* output,    // Output (H*W*3)
    const float* matrix_3x3_data, // 3x3 matrix in row-major order
    const int width,
    const int height)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const int idx = (pos.y * width + pos.x) * 3;
    float3 input_px = make_float3(input[idx], input[idx + 1], input[idx + 2]);

    // Create 3x3 matrix from data
    float3x3 transform_matrix_3x3(matrix_3x3_data);
    float3 output_px = color_transform_3x3(input_px, transform_matrix_3x3);

    output[idx] = output_px.x;
    output[idx + 1] = output_px.y;
    output[idx + 2] = output_px.z;
}

// PyTorch wrapper for generic 3x3 matrix transform
torch::Tensor color_transform_3x3(const torch::Tensor& input, const torch::Tensor& matrix_3x3) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(matrix_3x3.dtype() == torch::kFloat32, "Matrix must be float32");
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 3, "Input must be (H, W, 3)");
    TORCH_CHECK(matrix_3x3.dim() == 2 && matrix_3x3.size(0) == 3 && matrix_3x3.size(1) == 3, "Matrix must be (3, 3)");
    TORCH_CHECK(input.is_cuda() && matrix_3x3.is_cuda(), "Inputs must be on CUDA device");
    TORCH_CHECK(input.is_contiguous() && matrix_3x3.is_contiguous(), "Input tensors must be contiguous");

    const int height = input.size(0);
    const int width = input.size(1);

    auto output = torch::empty_like(input);

    dim3 block = block_size_2d;
    dim3 grid = grid2d(width, height);

    matrix_3x3_transform_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        matrix_3x3.data_ptr<float>(),
        width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in color_transform_3x3: ") + cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA synchronization error in color_transform_3x3: ") + cudaGetErrorString(err));
    }

    return output;
}




