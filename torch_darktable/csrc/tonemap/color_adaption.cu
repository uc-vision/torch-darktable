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
      float3 rgb = float3_load(image, pixel_idx);
      
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
  float3 rgb = float3_load(image, pixel_idx);
  
  // Scale to [0,1] range
  float range = bounds[1] - bounds[0] + eps;
  float3 scaled = (rgb - bounds[0]) / range;
  
  float gray = rgb_to_gray(scaled);
  float log_gray = logf(fmaxf(gray, min_gray));
  
  // Reduce + atomic writes in one step
  reduce_add(log_gray, &metrics[4]);     // log_mean
  reduce_add(gray, &metrics[5]);         // mean  
  reduce_add(scaled.x, &metrics[6]);     // rgb_mean.r
  reduce_add(scaled.y, &metrics[7]);     // rgb_mean.g
  reduce_add(scaled.z, &metrics[8]);     // rgb_mean.b
  reduce_min(log_gray, &metrics[2]);     // log_bounds_min
  reduce_max(log_gray, &metrics[3]);     // log_bounds_max
}


__global__ void compute_transform_kernel(
    const float* __restrict__ metrics,
    ColorTransform* __restrict__ transform_out,
    float intensity,
    float light_adapt,
    const float min_midtone = 0.3f,
    const float midtone_range = 0.7f,
    const float midtone_gamma = 1.4f,
    const float eps = 1e-6f
) {
    // --- Extract metrics ---
    float bounds_min_val = metrics[0];
    float bounds_max_val = metrics[1];
    float log_bounds_min = metrics[2];
    float log_bounds_max = metrics[3];
    float log_mean       = metrics[4];
    float linear_mean    = metrics[5];
    float3 rgb_mean      = make_float3(metrics[6], metrics[7], metrics[8]);
    float gray_mean      = linear_mean;  // use linear mean as grayscale mean

    // --- Key / map_key ---
    float key = (log_bounds_max - log_mean) / (log_bounds_max - log_bounds_min + eps);
    float map_key = min_midtone + midtone_range * powf(key, midtone_gamma);

    // --- Compute global adapted mean ---
    float3 gray_mean3 = make_float3(gray_mean, gray_mean, gray_mean);
    float3 adapt_mean = lerp(light_adapt, rgb_mean, gray_mean3);


    // --- Fill precompute struct ---
    transform_out->adapt_mean = adapt_mean;
    transform_out->map_key = map_key;

    transform_out->exposure = expf(intensity);

    transform_out->bounds_min = bounds_min_val;
    transform_out->range = bounds_max_val - bounds_min_val + eps;
}

void metrics_to_transform(ColorTransform& transform_out, const torch::Tensor& metrics, const TonemapParams& params, cudaStream_t stream) {

  TORCH_CHECK(metrics.dtype() == torch::kFloat32 && metrics.numel() == 9,
              "metrics must be float32 with 9 elements");
  TORCH_CHECK(metrics.device().is_cuda(), "metrics must be on CUDA device");

  ColorTransform* d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(ColorTransform)));


  compute_transform_kernel<<<1, 1, 0, stream>>>(
      metrics.data_ptr<float>(),
      d_tmp,
      params.intensity,
      params.light_adapt);

  CUDA_CHECK_KERNEL();


  CUDA_CHECK(cudaMemcpyToSymbolAsync(transform_out, d_tmp, sizeof(ColorTransform), 0, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaFree(d_tmp));
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
  auto metrics = torch::tensor({0.0f, 1.0f, FLT_MAX, -FLT_MAX, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, opts);

  int total_pixels = 0;
  torch::Tensor bounds = torch::tensor({0.0f, 1.0f}, opts);
  if (rescale) {
    bounds = compute_image_bounds(images, stride);
    metrics.slice(0, 0, 2) = bounds;
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
        height,
        width,
        stride,
        min_gray);
    CUDA_CHECK_KERNEL();
  }

  float norm_factor = 1.0f / (float)total_pixels;
  metrics.slice(0, 4, 9) *= norm_factor;
  return metrics;
}


