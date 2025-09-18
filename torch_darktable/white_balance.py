"""White balance estimation and application for Bayer images."""

from beartype import beartype
import torch

from .debayer import BayerPattern
from .extension import extension


@beartype
def apply_white_balance(bayer_image: torch.Tensor, gains: torch.Tensor, pattern: BayerPattern) -> torch.Tensor:
  """
  Apply white balance gains to a Bayer image.

  Args:
      bayer_image: Input Bayer image tensor (H, W)
      gains: White balance gains tensor (3,) for [R, G, B]
      pattern: Bayer pattern of the input image

  Returns:
      White-balanced Bayer image tensor with gains applied
  """
  return extension.apply_white_balance(bayer_image, gains, pattern.value)


@beartype
def estimate_white_balance(
  bayer_images: list[torch.Tensor],
  pattern: BayerPattern,
  quantile: float = 0.98,
  stride: int = 8,
) -> torch.Tensor:
  """
  Estimate white balance gains from multiple Bayer images.

  This function analyzes the brightest pixels (above the specified quantile)
  across multiple Bayer images to estimate proper white balance gains.

  Args:
      bayer_images: List of Bayer image tensors, all with same dimensions (H, W)
      pattern: Bayer pattern of the input images
      quantile: Quantile threshold for bright pixel selection (0.0-1.0)
      stride: Pixel sampling stride for efficiency

  Returns:
      White balance gains tensor (3,) for [R, G, B] channels

  Notes:
      - Pixels >= 1.0 (saturated) are automatically excluded
      - Only pixels above the intensity quantile are used for estimation
      - RGB values are estimated from 2x2 Bayer cells
      - Green channel is used as reference (gain = 1.0)
  """
  return extension.estimate_white_balance(bayer_images, pattern.value, quantile, stride)


__all__ = ['apply_white_balance', 'estimate_white_balance']
