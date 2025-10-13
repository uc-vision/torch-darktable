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
      vibrance: Vibrance adjustment (default 0.0)
  """

  gamma: float = 1.0
  intensity: float = 0.0
  light_adapt: float = 0.8
  vibrance: float = 0.0

  def to_cpp(self) -> extension.TonemapParams:
    """Convert to C++ TonemapParams struct."""
    return extension.TonemapParams(self.gamma, self.intensity, self.light_adapt, self.vibrance)

  @classmethod
  def from_cpp(cls, cpp_params: extension.TonemapParams) -> 'TonemapParameters':
    """Create from C++ TonemapParams struct."""
    return cls(cpp_params.gamma, cpp_params.intensity, cpp_params.light_adapt, cpp_params.vibrance)


@beartype
def metrics_to_dict(metrics: torch.Tensor) -> dict[str, float | tuple[float, float, float]]:
  """Convert 5-element metrics tensor to named dictionary."""
  assert metrics.numel() == 5, f'Expected 5 elements, got {metrics.numel()}'
  m = metrics.cpu().numpy()
  return {
    'log_mean': float(m[0]),
    'linear_mean': float(m[1]),
    'rgb_mean': (float(m[2]), float(m[3]), float(m[4])),
  }


@beartype
def metrics_from_dict(
  metrics_dict: dict[str, float | tuple[float, float, float]], device: torch.device = torch.device('cuda')
) -> torch.Tensor:
  """Convert named dictionary to 5-element metrics tensor."""
  rgb_mean = metrics_dict['rgb_mean']
  assert isinstance(rgb_mean, tuple), 'RGB mean must be a tuple'

  return torch.tensor(
    [
      metrics_dict['log_mean'],
      metrics_dict['linear_mean'],
      rgb_mean[0],  # r
      rgb_mean[1],  # g
      rgb_mean[2],  # b
    ],
    device=device,
    dtype=torch.float32,
  )


@beartype
def print_metrics(metrics: torch.Tensor):
  """Print metrics in a nicely formatted way."""
  d = metrics_to_dict(metrics)
  print('Image Metrics:')
  rgb = d['rgb_mean']
  assert isinstance(rgb, tuple), 'RGB mean must be a tuple'

  print(f'  Log Mean: {d["log_mean"]:.4f}')
  print(f'  Linear Mean: {d["linear_mean"]:.4f}')
  print(f'  RGB Mean: ({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})')


@beartype
def reinhard_tonemap(image: torch.Tensor, metrics: torch.Tensor, params: TonemapParameters) -> torch.Tensor:
  """
  Apply Reinhard tone mapping to HDR image.

  Args:
      image: Input RGB image tensor of shape (H, W, 3), float32
      metrics: 7-element tensor of image statistics
      params: TonemapParameters object with tone mapping settings

  Returns:
      Tone mapped image as uint8 tensor (H, W, 3) in range [0, 255]
  """
  assert image.dim() == 3 and image.size(2) == 3, 'Input must be (H, W, 3)'
  assert image.dtype == torch.float32, 'Input must be float32'
  assert image.device.type == 'cuda', 'Input must be on CUDA device'
  assert metrics.numel() == 5, 'Metrics tensor must have 5 elements'

  return extension.reinhard_tonemap(image, metrics, params.to_cpp())


@beartype
def aces_tonemap(image: torch.Tensor, params: TonemapParameters, metrics: torch.Tensor | None = None) -> torch.Tensor:
  """
  Apply ACES tone mapping (industry standard).

  Args:
      image: Input RGB image tensor of shape (H, W, 3), float32
      params: TonemapParameters object with tone mapping settings
      metrics: Optional 7-element tensor of image statistics for adaptive tone mapping

  Returns:
      Tone mapped image as uint8 tensor (H, W, 3) in range [0, 255]
  """
  assert image.dim() == 3 and image.size(2) == 3, 'Input must be (H, W, 3)'
  assert image.dtype == torch.float32, 'Input must be float32'
  assert image.device.type == 'cuda', 'Input must be on CUDA device'

  if metrics is not None:
    assert metrics.numel() == 5 and metrics.dtype == torch.float32 and metrics.device.type == 'cuda'
    return extension.adaptive_aces_tonemap(image, metrics, params.to_cpp())
  return extension.aces_tonemap(image, params.to_cpp())


@beartype
def linear_tonemap(image: torch.Tensor, metrics: torch.Tensor, params: TonemapParameters) -> torch.Tensor:
  """
  Apply linear tone mapping with simple normalization.

  Simple linear tonemap that normalizes by max value and applies gamma correction.

  Args:
      image: Input linear HDR image tensor (H, W, 3), float32, CUDA
      metrics: 7-element tensor of image statistics
      params: TonemapParameters object with tone mapping settings

  Returns:
      Tonemapped image as uint8 tensor (H, W, 3) in range [0, 255]
  """
  assert image.dim() == 3 and image.size(2) == 3, 'Input must be (H, W, 3)'
  assert image.dtype == torch.float32, 'Input must be float32'
  assert image.device.type == 'cuda', 'Input must be on CUDA device'
  assert metrics.numel() == 5 and metrics.dtype == torch.float32 and metrics.device.type == 'cuda'

  return extension.linear_tonemap(image, metrics, params.to_cpp())


compute_image_bounds = beartype(extension.compute_image_bounds)


@beartype
def compute_image_metrics(
  images: list[torch.Tensor], stride: int = 8, min_gray: float = 1e-4, rescale: bool = False
) -> torch.Tensor:
  return extension.compute_image_metrics(images, stride, min_gray, rescale)


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
