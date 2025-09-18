"""Tone mapping algorithms and utilities."""

from dataclasses import dataclass

from beartype import beartype
import torch

from .extension import extension


@dataclass(frozen=True)
class TonemapParameters:
  """Parameters for tone mapping algorithms.

  This is a convenient Python dataclass that can be converted to/from
  the C++ TonemapParams struct for efficient processing.

  Args:
      gamma: Gamma correction factor (default 1.0)
      intensity: Exposure adjustment in stops (default 0.0)
      light_adapt: Local vs global adaptation blend, 0=global, 1=local (default 0.8)
  """

  gamma: float = 1.0
  intensity: float = 0.0
  light_adapt: float = 0.8

  def to_cpp(self) -> extension.TonemapParams:
    """Convert to C++ TonemapParams struct."""
    return extension.TonemapParams(self.gamma, self.intensity, self.light_adapt)

  @classmethod
  def from_cpp(cls, cpp_params: extension.TonemapParams) -> 'TonemapParameters':
    """Create from C++ TonemapParams struct."""
    return cls(cpp_params.gamma, cpp_params.intensity, cpp_params.light_adapt)


@beartype
def metrics_to_dict(metrics: torch.Tensor) -> dict[str, float]:
  """Convert 9-element metrics tensor to named dictionary."""
  assert metrics.numel() == 9, f'Expected 9 elements, got {metrics.numel()}'
  m = metrics.cpu().numpy()
  return {
    'bounds_min': float(m[0]),
    'bounds_max': float(m[1]),
    'log_bounds_min': float(m[2]),
    'log_bounds_max': float(m[3]),
    'log_mean': float(m[4]),
    'mean': float(m[5]),
    'rgb_mean_r': float(m[6]),
    'rgb_mean_g': float(m[7]),
    'rgb_mean_b': float(m[8]),
  }


@beartype
def metrics_from_dict(metrics_dict: dict[str, float], device: torch.device = torch.device('cuda')) -> torch.Tensor:
  """Convert named dictionary to 9-element metrics tensor."""
  return torch.tensor(
    [
      metrics_dict['bounds_min'],
      metrics_dict['bounds_max'],
      metrics_dict['log_bounds_min'],
      metrics_dict['log_bounds_max'],
      metrics_dict['log_mean'],
      metrics_dict['mean'],
      metrics_dict['rgb_mean_r'],
      metrics_dict['rgb_mean_g'],
      metrics_dict['rgb_mean_b'],
    ],
    device=device,
    dtype=torch.float32,
  )


@beartype
def print_metrics(metrics: torch.Tensor):
  """Print metrics in a nicely formatted way."""
  d = metrics_to_dict(metrics)
  print('Reinhard Image Metrics:')
  print(f'  Bounds: [{d["bounds_min"]:.4f}, {d["bounds_max"]:.4f}]')
  print(f'  Log Bounds: [{d["log_bounds_min"]:.4f}, {d["log_bounds_max"]:.4f}]')
  print(f'  Log Mean: {d["log_mean"]:.4f}')
  print(f'  Mean: {d["mean"]:.4f}')
  print(f'  RGB Mean: ({d["rgb_mean_r"]:.4f}, {d["rgb_mean_g"]:.4f}, {d["rgb_mean_b"]:.4f})')


@beartype
def reinhard_tonemap(image: torch.Tensor, metrics: torch.Tensor, params: TonemapParameters) -> torch.Tensor:
  """
  Apply Reinhard tone mapping to HDR image.

  Args:
      image: Input RGB image tensor of shape (H, W, 3), float32
      metrics: 9-element tensor of image statistics
      params: TonemapParameters object with tone mapping settings

  Returns:
      Tone mapped image as uint8 tensor (H, W, 3) in range [0, 255]
  """
  assert image.dim() == 3 and image.size(2) == 3, 'Input must be (H, W, 3)'
  assert image.dtype == torch.float32, 'Input must be float32'
  assert image.device.type == 'cuda', 'Input must be on CUDA device'
  assert metrics.numel() == 9, 'Metrics tensor must have 9 elements'

  return extension.reinhard_tonemap(image, metrics, params.to_cpp())


@beartype
def aces_tonemap(image: torch.Tensor, metrics: torch.Tensor, params: TonemapParameters) -> torch.Tensor:
  """
  Apply ACES tone mapping (industry standard).

  Note: This doesn't use any image statistics, so exposure may need manual adjustment.

  Args:
      image: Input RGB image tensor of shape (H, W, 3), float32
      metrics: 9-element tensor of image statistics
      params: TonemapParameters object with tone mapping settings

  Returns:
      Tone mapped image as uint8 tensor (H, W, 3) in range [0, 255]
  """
  assert image.dim() == 3 and image.size(2) == 3, 'Input must be (H, W, 3)'
  assert image.dtype == torch.float32, 'Input must be float32'
  assert image.device.type == 'cuda', 'Input must be on CUDA device'
  assert metrics.numel() == 9 and metrics.dtype == torch.float32 and metrics.device.type == 'cuda'

  return extension.aces_tonemap(image, metrics, params.to_cpp())


@beartype
def linear_tonemap(image: torch.Tensor, metrics: torch.Tensor, params: TonemapParameters) -> torch.Tensor:
  """
  Apply linear tone mapping with simple normalization.

  Simple linear tonemap that normalizes by max value and applies gamma correction.

  Args:
      image: Input linear HDR image tensor (H, W, 3), float32, CUDA
      metrics: 9-element tensor of image statistics
      params: TonemapParameters object with tone mapping settings

  Returns:
      Tonemapped image as uint8 tensor (H, W, 3)
  """
  assert image.dim() == 3 and image.size(2) == 3, 'Input must be (H, W, 3)'
  assert image.dtype == torch.float32, 'Input must be float32'
  assert image.device.type == 'cuda', 'Input must be on CUDA device'
  assert metrics.numel() == 9 and metrics.dtype == torch.float32 and metrics.device.type == 'cuda'

  return extension.linear_tonemap(image, metrics, params.to_cpp())


compute_image_bounds = beartype(extension.compute_image_bounds)


@beartype
def compute_image_metrics(images: list[torch.Tensor], stride: int = 8, min_gray: float = 1e-4) -> torch.Tensor:
  return extension.compute_image_metrics(images, stride, min_gray)


__all__ = [
  'TonemapParameters',
  'aces_tonemap',
  'compute_image_bounds',
  'compute_image_metrics',
  'linear_tonemap',
  'metrics_from_dict',
  'metrics_to_dict',
  'print_metrics',
  'reinhard_tonemap',
]
