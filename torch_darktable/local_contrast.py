"""Local contrast enhancement algorithms."""

from dataclasses import dataclass

from beartype import beartype
import torch

from .extension import extension


@beartype
@dataclass
class LaplacianParams:
  """Parameters for local Laplacian filtering."""

  num_gamma: int = 6
  sigma: float = 0.2
  shadows: float = 1.0
  highlights: float = 1.0
  clarity: float = 0.0


class Laplacian:
  """Laplacian with shape validation."""

  @beartype
  def __init__(self, device: torch.device, image_size: tuple[int, int], params: LaplacianParams):
    """Create a local Laplacian filter object.

    Args:
        device: CUDA device to use
        image_size: (width, height) of the image
        params: Laplacian filter parameters
    """
    width, height = image_size
    self._laplacian = extension.Laplacian(
      device,
      width,
      height,
      params.num_gamma,
      params.sigma,
      params.shadows,
      params.highlights,
      params.clarity,
    )

  def process(self, input_tensor: torch.Tensor) -> torch.Tensor:
    expected_shape = (self._laplacian.height, self._laplacian.width)
    if input_tensor.shape != expected_shape:
      raise RuntimeError(f'Laplacian input shape {input_tensor.shape} != expected {expected_shape}')
    return self._laplacian.process(input_tensor)

  @beartype
  def process_rgb(self, input_image: torch.Tensor) -> torch.Tensor:
    """Apply local Laplacian filtering to RGB image."""
    luminance = extension.compute_luminance(input_image)
    return extension.modify_luminance(input_image, self.process(luminance))

  @property
  def image_size(self) -> tuple[int, int]:
    return (self._laplacian.width, self._laplacian.height)

  @property
  def sigma(self) -> float:
    return self._laplacian.sigma

  @property
  def shadows(self) -> float:
    return self._laplacian.shadows

  @property
  def highlights(self) -> float:
    return self._laplacian.highlights

  @property
  def clarity(self) -> float:
    return self._laplacian.clarity


class Bilateral:
  """Bilateral with shape validation."""

  @beartype
  def __init__(
    self,
    device: torch.device,
    image_size: tuple[int, int],
    *,
    sigma_s: float,
    sigma_r: float,
  ):
    """Create a bilateral filter object.

    Args:
        device: CUDA device to use
        image_size: (width, height) of the image
        sigma_s: Spatial standard deviation
        sigma_r: Luminance range standard deviation
    """
    width, height = image_size
    self._bilateral = extension.Bilateral(device, width, height, sigma_s, sigma_r)

  def process(self, luminance: torch.Tensor, detail: float) -> torch.Tensor:
    expected_shape = (self._bilateral.height, self._bilateral.width)
    if luminance.shape != expected_shape:
      raise RuntimeError(f'Bilateral input shape {luminance.shape} != expected {expected_shape}')
    return self._bilateral.process(luminance, detail)

  @beartype
  def process_rgb(self, input_image: torch.Tensor, detail: float) -> torch.Tensor:
    """Apply bilateral filtering to RGB image."""
    assert input_image.dim() == 3, f'image must have 3 dimensions, got {input_image.shape}'
    luminance = extension.compute_luminance(input_image)
    return extension.modify_luminance(input_image, self.process(luminance, float(detail)))

  @beartype
  def process_log_rgb(
    self,
    input_image: torch.Tensor,
    detail: float,
    eps: float = 1e-6,
  ) -> torch.Tensor:
    """Apply bilateral filtering to RGB image in log space."""
    log_luminance = extension.compute_log_luminance(input_image, eps)
    return extension.modify_log_luminance(input_image, self.process(log_luminance, float(detail)), eps)

  @property
  def image_size(self) -> tuple[int, int]:
    return (self._bilateral.width, self._bilateral.height)

  @property
  def sigma_s(self) -> float:
    return self._bilateral.sigma_s

  @property
  def sigma_r(self) -> float:
    return self._bilateral.sigma_r


__all__ = [
  'Bilateral',
  'Laplacian',
  'LaplacianParams',
]
