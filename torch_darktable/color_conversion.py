"""Color space conversion functions."""

import torch
from .extension import extension
from beartype import beartype


@beartype
def compute_luminance(rgb_image: torch.Tensor) -> torch.Tensor:
  """
  Compute luminance from RGB image.

  Args:
      rgb_image: Input RGB image tensor

  Returns:
      Luminance tensor
  """
  return extension.compute_luminance(rgb_image)


@beartype
def modify_luminance(rgb_image: torch.Tensor, luminance_multiplier: torch.Tensor) -> torch.Tensor:
  """
  Modify luminance of RGB image while preserving chromaticity.

  Args:
      rgb_image: Input RGB image tensor
      luminance_multiplier: Luminance scaling factor tensor

  Returns:
      Modified RGB image tensor
  """
  return extension.modify_luminance(rgb_image, luminance_multiplier)


@beartype
def compute_log_luminance(rgb_image: torch.Tensor, eps: float) -> torch.Tensor:
  """
  Compute log luminance from RGB image with clamping to avoid NaN.

  Args:
      rgb_image: Input RGB image tensor
      eps: Small value to clamp lower side to avoid NaN

  Returns:
      Log luminance tensor
  """
  return extension.compute_log_luminance(rgb_image, eps)


@beartype
def modify_log_luminance(rgb_image: torch.Tensor, log_luminance: torch.Tensor, eps: float) -> torch.Tensor:
  """
  Update RGB image with modified log luminance.

  Args:
      rgb_image: Input RGB image tensor
      log_luminance: Modified log luminance tensor

  Returns:
      RGB image tensor with updated log luminance
  """
  return extension.modify_log_luminance(rgb_image, log_luminance, eps)


@beartype
def modify_hsl(
  rgb_image: torch.Tensor,
  hue_adjust: float = 0.0,
  sat_adjust: float = 0.0,
  lum_adjust: float = 0.0,
) -> torch.Tensor:
  """
  Update RGB image with modified hue, saturation, and luminance.

  Args:
      rgb_image: Input RGB image tensor of shape (H, W, 3) with values in [0, 1]
      hue_adjust: Hue adjustment in [0, 1] range (0.0 = no change)
      sat_adjust: Saturation adjustment (0.0 = no change, negative = less saturated)
      lum_adjust: Luminance adjustment (0.0 = no change, negative = darker)

  Returns:
      Modified RGB image tensor of shape (H, W, 3)
  """
  return extension.modify_hsl(rgb_image, hue_adjust, sat_adjust, lum_adjust)


@beartype
def modify_vibrance(rgb_image: torch.Tensor, amount: float = 0.0) -> torch.Tensor:
  """
  Update RGB image with darktable-style vibrance adjustment.

  Vibrance is a perceptually-aware saturation enhancement that works in LAB color space.
  It enhances colorful areas more than neutral areas, resulting in more natural-looking
  color enhancement compared to HSL saturation.

  Args:
      rgb_image: Input RGB image tensor of shape (H, W, 3) with values in [0, 1]
      amount: Vibrance adjustment amount (0.0 = no change, positive = more vibrant)

  Returns:
      Modified RGB image tensor of shape (H, W, 3)
  """
  return extension.modify_vibrance(rgb_image, amount)


@beartype
def rgb_to_lab(rgb_image: torch.Tensor) -> torch.Tensor:
  """
  Convert RGB to CIELAB color space.

  Args:
      rgb_image: Input RGB image tensor

  Returns:
      LAB image tensor
  """
  return extension.rgb_to_lab(rgb_image)


@beartype
def lab_to_rgb(lab_image: torch.Tensor) -> torch.Tensor:
  """
  Convert CIELAB to RGB color space.

  Args:
      lab_image: Input LAB image tensor

  Returns:
      RGB image tensor
  """
  return extension.lab_to_rgb(lab_image)


@beartype
def rgb_to_xyz(rgb_image: torch.Tensor) -> torch.Tensor:
  """
  Convert RGB to CIE XYZ color space.

  Args:
      rgb_image: Input RGB image tensor

  Returns:
      XYZ image tensor
  """
  return extension.rgb_to_xyz(rgb_image)


@beartype
def xyz_to_lab(xyz_image: torch.Tensor) -> torch.Tensor:
  """
  Convert CIE XYZ to CIELAB color space.

  Args:
      xyz_image: Input XYZ image tensor

  Returns:
      LAB image tensor
  """
  return extension.xyz_to_lab(xyz_image)


@beartype
def lab_to_xyz(lab_image: torch.Tensor) -> torch.Tensor:
  """
  Convert CIELAB to CIE XYZ color space.

  Args:
      lab_image: Input LAB image tensor

  Returns:
      XYZ image tensor
  """
  return extension.lab_to_xyz(lab_image)


@beartype
def xyz_to_rgb(xyz_image: torch.Tensor) -> torch.Tensor:
  """
  Convert CIE XYZ to RGB color space.

  Args:
      xyz_image: Input XYZ image tensor

  Returns:
      RGB image tensor
  """
  return extension.xyz_to_rgb(xyz_image)


@beartype
def color_transform_3x3(image: torch.Tensor, matrix_3x3: torch.Tensor) -> torch.Tensor:
  """
  Apply a 3x3 color transformation matrix to an RGB image with clamping to [0,1].

  Args:
      image: Input RGB image tensor (H, W, 3)
      matrix_3x3: 3x3 transformation matrix (3, 3)

  Returns:
      Transformed RGB image tensor with values clamped to [0,1]
  """
  return extension.color_transform_3x3(image, matrix_3x3)


__all__ = [
  'compute_luminance',
  'modify_luminance',
  'compute_log_luminance',
  'modify_log_luminance',
  'modify_hsl',
  'modify_vibrance',
  'rgb_to_lab',
  'lab_to_rgb',
  'rgb_to_xyz',
  'xyz_to_lab',
  'lab_to_xyz',
  'xyz_to_rgb',
  'color_transform_3x3',
]
