#include "color_adaption.h"
#include "../cuda_utils.h"
#include "../device_math.h"
#include "../reduction.h"
#include <float.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

// Image metrics computation functions
// Compute image bounds kernel using cooperative groups
__global__ void compute_bounds_kernel(
  const float* __restrict__ image,
  float* __restrict__ bounds,
  int height,
  int width,
  int stride
) {
  
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
  int y = (blockIdx.y * blockDim.y + threadIdx.y) * stride;

  // Each thread processes one pixel
  if (x < width && y < height) {
      // RGB image: (height, width, 3)
      int pixel_idx = y * width + x;
      float3 rgb = load<float3>(image, pixel_idx);
      
      float local_min = fminf(fminf(rgb.x, rgb.y), rgb.z);
      float local_max = fmaxf(fmaxf(rgb.x, rgb.y), rgb.z);

      // Reduce + atomic write in one step
      reduce_min(local_min, &bounds[0]);
      reduce_max(local_max, &bounds[1]);
  }
}

// Compute image metrics kernel
__global__ void compute_metrics_kernel(
  const float* __restrict__ image,
  float* __restrict__ metrics,
  float* __restrict__ bounds,
  float* __restrict__ valid_count,
  int height,
  int width,
  int stride,
  float min_gray,
  const float eps = 1e-6f
) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
  int y = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
  
  if (x >= width || y >= height) return;
  
  // Load RGB pixel
  int pixel_idx = y * width + x;
  float3 rgb = load<float3>(image, pixel_idx);
  
  // Scale to [0,1] range
  float range = bounds[1] - bounds[0] + eps;
  float3 scaled = (rgb - bounds[0]) / range;
  
  // Check if pixel is saturated (any component >= 0.98)
  const float saturation_threshold = 0.98f;
  bool is_saturated = (scaled.x >= saturation_threshold || 
                       scaled.y >= saturation_threshold || 
                       scaled.z >= saturation_threshold);
  
  // Use mask to zero out saturated pixels instead of branching
  float mask = is_saturated ? 0.0f : 1.0f;
  
  float gray = rgb_to_gray(scaled);
  float log_gray = logf(fmaxf(gray, min_gray));
  
  // Reduce + atomic writes with masking (log bounds removed - fixed values)
  reduce_add(log_gray * mask, &metrics[0]);     // log_mean
  reduce_add(gray * mask, &metrics[1]);         // linear_mean  
  reduce_add(scaled.x * mask, &metrics[2]);     // rgb_mean.r
  reduce_add(scaled.y * mask, &metrics[3]);     // rgb_mean.g
  reduce_add(scaled.z * mask, &metrics[4]);     // rgb_mean.b
  
  // Count valid pixels
  reduce_add(mask, valid_count);         // valid_pixel_count
}





torch::Tensor compute_image_bounds(const std::vector<torch::Tensor>& images, int stride) {
  TORCH_CHECK(!images.empty(), "images must be non-empty");

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(images[0].device());
  auto bounds = torch::tensor({FLT_MAX, -FLT_MAX}, opts);

  for (const auto& img : images) {
    check_image(img);
    
    int height = (int)img.size(0);
    int width = (int)img.size(1);

    int sample_width = (width + stride - 1) / stride;
    int sample_height = (height + stride - 1) / stride;

    const dim3 block_size(16, 16);
    const dim3 grid_size((sample_width + block_size.x - 1) / block_size.x,
                         (sample_height + block_size.y - 1) / block_size.y);

    compute_bounds_kernel<<<grid_size, block_size>>>(
        img.data_ptr<float>(),
        bounds.data_ptr<float>(),
        height,
        width,
        stride);
 
 }

  CUDA_CHECK_KERNEL();
  return bounds;
}

torch::Tensor compute_image_metrics(const std::vector<torch::Tensor>& images, int stride, 
    float min_gray, bool rescale) {
  TORCH_CHECK(!images.empty(), "images must be non-empty");

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(images[0].device());
  auto metrics = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, opts);
  auto valid_count = torch::zeros(1, opts);

  int total_pixels = 0;
  torch::Tensor bounds = torch::tensor({0.0f, 1.0f}, opts);
  if (rescale) {
    bounds = compute_image_bounds(images, stride);
    // bounds are handled externally now
  } 

  for (const auto& img : images) {
    int height = (int)img.size(0);
    int width = (int)img.size(1);

    int sample_width = (width + stride - 1) / stride;
    int sample_height = (height + stride - 1) / stride;
    total_pixels += sample_width * sample_height;

    const dim3 block_size(16, 16);
    const dim3 grid_size((sample_width + block_size.x - 1) / block_size.x,
                         (sample_height + block_size.y - 1) / block_size.y);

    compute_metrics_kernel<<<grid_size, block_size>>>(
        img.data_ptr<float>(),
        metrics.data_ptr<float>(),
        bounds.data_ptr<float>(),
        valid_count.data_ptr<float>(),
        height,
        width,
        stride,
        min_gray);
    CUDA_CHECK_KERNEL();
  }

  // Use valid pixel count for normalization instead of total pixels
  float valid_pixels = valid_count.item<float>();
  float norm_factor = 1.0f / fmaxf(valid_pixels, 1.0f);
  metrics *= norm_factor;
  return metrics;
}


