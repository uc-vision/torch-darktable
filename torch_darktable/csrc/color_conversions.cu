#include <cuda_runtime.h>
#include "color_conversions.h"
#include "cuda_utils.h"
#include "device_math.h"
#include "device_conversions.h"

// ============================================================================
// PATTERN 1: Color Space Conversion (float3 -> float3)
// ============================================================================

// Converter structs for color space conversions
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

struct ConvertWithMatrix3x3 {
    float3x3 matrix;
    
    __host__ __device__ ConvertWithMatrix3x3(const float3x3& m) : matrix(m) {}
    
    __device__ float3 operator()(float3 input) const {
        return color_transform_3x3(input, matrix);
    }
};

struct AdjustHSL {
    float hue_adjust;
    float sat_adjust;
    float lum_adjust;
    __device__ float3 operator()(float3 rgb) const {
        return modify_rgb_hsl(rgb, hue_adjust, sat_adjust, lum_adjust);
    }
};

struct AdjustVibrance {
    float amount;
    __device__ float3 operator()(float3 rgb) const {
        return modify_rgb_vibrance_dt(rgb, amount);
    }
};




// Generic kernel for color space conversion
template <typename Converter>
__global__ void convert_color_kernel(
    float* input,     // Input (H*W*3)
    float* output,    // Output (H*W*3)
    const int width,
    const int height,
    Converter converter)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const int pixel_idx = pos.y * width + pos.x;
    float3 input_px = load<float3>(input, pixel_idx);
    float3 output_px = converter(input_px);
    store(output_px, output, pixel_idx);
}

// Generic wrapper for color space conversion
template <typename Converter>
torch::Tensor convert_color(const torch::Tensor& input, Converter converter, const char* kernel_name) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 3, "Input must be (H, W, 3)");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const int height = input.size(0);
    const int width = input.size(1);
    auto output = torch::empty_like(input);

    dim3 block = block_size_2d;
    dim3 grid = grid2d(width, height);

    convert_color_kernel<Converter><<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), width, height, converter);

    CUDA_CHECK_KERNEL();
    return output;
}

// Color space conversion wrappers
torch::Tensor rgb_to_xyz(const torch::Tensor& rgb) {
    return convert_color(rgb, ConvertRgbToXyz{}, "rgb_to_xyz");
}

torch::Tensor xyz_to_lab(const torch::Tensor& xyz) {
    return convert_color(xyz, ConvertXyzToLab{}, "xyz_to_lab");
}

torch::Tensor lab_to_xyz(const torch::Tensor& lab) {
    return convert_color(lab, ConvertLabToXyz{}, "lab_to_xyz");
}

torch::Tensor xyz_to_rgb(const torch::Tensor& xyz) {
    return convert_color(xyz, ConvertXyzToRgb{}, "xyz_to_rgb");
}

torch::Tensor rgb_to_lab(const torch::Tensor& rgb) {
    return convert_color(rgb, ConvertRgbToLab{}, "rgb_to_lab");
}

torch::Tensor lab_to_rgb(const torch::Tensor& lab) {
    return convert_color(lab, ConvertLabToRgb{}, "lab_to_rgb");
}


torch::Tensor modify_hsl(const torch::Tensor& rgb, float hue_adjust, float sat_adjust, float lum_adjust) {
    return convert_color(rgb, AdjustHSL{hue_adjust, sat_adjust, lum_adjust}, "adjust_hsl");
}

torch::Tensor modify_vibrance(const torch::Tensor& rgb, float amount) {
    return convert_color(rgb, AdjustVibrance{amount}, "adjust_vibrance");
}



torch::Tensor color_transform_3x3(const torch::Tensor& input, const torch::Tensor& matrix_3x3) {
    TORCH_CHECK(matrix_3x3.dtype() == torch::kFloat32, "Matrix must be float32");
    TORCH_CHECK(matrix_3x3.dim() == 2 && matrix_3x3.size(0) == 3 && matrix_3x3.size(1) == 3, "Matrix must be (3, 3)");
    TORCH_CHECK(matrix_3x3.is_cuda() && matrix_3x3.is_contiguous(), "Matrix tensor must be contiguous CUDA tensor");
    
    float* data = matrix_3x3.data_ptr<float>();
    float3x3 matrix(data);  // Now works on host thanks to __host__ __device__
    return convert_color(input, ConvertWithMatrix3x3{matrix}, "color_transform_3x3");
}

// ============================================================================
// PATTERN 2: Channel Extraction (float3 -> float)
// ============================================================================

// Converter structs for channel extraction
struct ExtractLuminance {
    __device__ float operator()(float3 input) const {
        return rgb_to_lab_l(clip(input));
    }
};

struct ExtractLogLuminance {
    float eps;
    
    __device__ ExtractLogLuminance(float epsilon = 1e-6f) : eps(epsilon) {}
    
    __device__ float operator()(float3 input) const {
        float lum = rgb_to_lab_l(clip(input));
        return logf(fmaxf(eps, lum));
    }
};

// Generic kernel for channel extraction
template <typename Converter>
__global__ void extract_channel_kernel(
    float* input,     // Input (H*W*3)
    float* output,    // Output (H*W*1)
    const int width,
    const int height,
    Converter converter)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const int pixel_idx = pos.y * width + pos.x;
    float3 input_px = load<float3>(input, pixel_idx);
    float result = converter(input_px);
    output[pixel_idx] = result;
}

// Generic wrapper for channel extraction
template <typename Converter>
torch::Tensor extract_channel(const torch::Tensor& input, Converter converter, const char* kernel_name) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 3, "Input must be (H, W, 3)");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const int height = input.size(0);
    const int width = input.size(1);
    auto output = torch::zeros({height, width}, input.options());

    dim3 block = block_size_2d;
    dim3 grid = grid2d(width, height);

    extract_channel_kernel<Converter><<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), width, height, converter);

    CUDA_CHECK_KERNEL();
    return output;
}

// Channel extraction wrappers
torch::Tensor compute_luminance(const torch::Tensor& rgb) {
    return extract_channel(rgb, ExtractLuminance{}, "compute_luminance");
}

torch::Tensor compute_log_luminance(const torch::Tensor& rgb, float eps) {
    TORCH_CHECK(eps > 0.0f, "Epsilon must be positive");
    return extract_channel(rgb, ExtractLogLuminance{eps}, "compute_log_luminance");
}

// ============================================================================
// PATTERN 3: Color Modification (float3, float -> float3)
// ============================================================================

// Converter structs for color modification
struct ModifyLuminance {
    __device__ float3 operator()(float3 rgb, float luminance) const {
        return modify_rgb_luminance(rgb, luminance);
    }
};

struct ModifyLogLuminance {
    float eps;
    
    __device__ ModifyLogLuminance(float epsilon) : eps(epsilon) {}
    
    __device__ float3 operator()(float3 rgb, float log_lum) const {
        return modify_rgb_log_luminance(rgb, log_lum, eps);
    }
};



// Generic kernel for color modification
template <typename Converter>
__global__ void modify_color_kernel(
    float* input1,        // Input 1 (H*W*3)
    float* input2,        // Input 2 (H*W*1) 
    float* output,        // Output (H*W*3)
    const int width,
    const int height,
    Converter converter)
{
    int2 pos = pixel_index();
    if(pos.x >= width || pos.y >= height) return;

    const int pixel_idx = pos.y * width + pos.x;
    float3 input1_px = load<float3>(input1, pixel_idx);
    float input2_val = input2[pixel_idx];
    float3 result = converter(input1_px, input2_val);
    store(result, output, pixel_idx);
}

// Generic wrapper for color modification
template <typename Converter>
torch::Tensor modify_color(const torch::Tensor& input1, const torch::Tensor& input2, Converter converter, const char* kernel_name) {
    TORCH_CHECK(input1.dtype() == torch::kFloat32, "Input1 must be float32");
    TORCH_CHECK(input2.dtype() == torch::kFloat32, "Input2 must be float32");
    TORCH_CHECK(input1.dim() == 3 && input1.size(2) == 3, "Input1 must be (H, W, 3)");
    TORCH_CHECK(input2.dim() == 2, "Input2 must be (H, W)");
    TORCH_CHECK(input1.is_cuda() && input2.is_cuda(), "Inputs must be on CUDA device");
    TORCH_CHECK(input1.is_contiguous() && input2.is_contiguous(), "Input tensors must be contiguous");

    const int height = input1.size(0);
    const int width = input1.size(1);
    TORCH_CHECK(input2.size(0) == height && input2.size(1) == width, 
                "Input dimensions must match");

    auto output = torch::empty_like(input1);

    dim3 block = block_size_2d;
    dim3 grid = grid2d(width, height);

    modify_color_kernel<Converter><<<grid, block>>>(
        input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(), 
        width, height, converter);

    CUDA_CHECK_KERNEL();
    return output;
}

// Color modification wrappers
torch::Tensor modify_luminance(const torch::Tensor& rgb, const torch::Tensor& new_luminance) {
    return modify_color(rgb, new_luminance, ModifyLuminance{}, "modify_luminance");
}

torch::Tensor modify_log_luminance(const torch::Tensor& rgb, const torch::Tensor& log_luminance, float eps) {
    TORCH_CHECK(eps > 0.0f, "Epsilon must be positive");
    return modify_color(rgb, log_luminance, ModifyLogLuminance{eps}, "modify_log_luminance");
}

