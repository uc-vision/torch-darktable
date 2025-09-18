"""Bayer demosaicing algorithms and utilities."""

from enum import Enum

from beartype import beartype
import torch

from .extension import extension


class BayerPattern(Enum):
  RGGB = extension.BayerPattern.RGGB
  BGGR = extension.BayerPattern.BGGR
  GRBG = extension.BayerPattern.GRBG
  GBRG = extension.BayerPattern.GBRG


class Packed12Format(Enum):
  STANDARD = 0
  IDS = 1


class Bilinear5x5:
  def __init__(self, bayer_pattern: BayerPattern):
    self.bayer_pattern = bayer_pattern

  def process(self, image: torch.Tensor) -> torch.Tensor:
    return bilinear5x5_demosaic(image, self.bayer_pattern)


def create_ppg(
  device: torch.device,
  image_size: tuple[int, int],
  bayer_pattern: BayerPattern,
  *,
  median_threshold: float = 0.0,
) -> extension.PPG:
  """
  Create a PPG demosaic object.
  """
  width, height = image_size
  return extension.PPG(device, width, height, bayer_pattern.value, median_threshold)


@beartype
def create_rcd(
  device: torch.device,
  image_size: tuple[int, int],
  bayer_pattern: BayerPattern,
  *,
  input_scale: float = 1.0,
  output_scale: float = 1.0,
) -> extension.RCD:
  """
  Create an RCD demosaic object.
  """
  width, height = image_size
  return extension.RCD(device, width, height, bayer_pattern.value, input_scale, output_scale)


@beartype
def create_bilinear(bayer_pattern: BayerPattern) -> Bilinear5x5:
  return Bilinear5x5(bayer_pattern)


@beartype
def create_postprocess(
  device: torch.device,
  image_size: tuple[int, int],
  bayer_pattern: BayerPattern,
  *,
  color_smoothing_passes: int = 0,
  green_eq_local: bool = False,
  green_eq_global: bool = False,
  green_eq_threshold: float = 0.04,
) -> extension.PostProcess:
  """
  Create a post-process object for demosaiced images.
  """
  width, height = image_size
  return extension.PostProcess(
    device,
    width,
    height,
    bayer_pattern.value,
    color_smoothing_passes,
    green_eq_local,
    green_eq_global,
    green_eq_threshold,
  )


# 12-bit packing/unpacking functions
@beartype
def encode12(image: torch.Tensor, format_type: Packed12Format = Packed12Format.STANDARD) -> torch.Tensor:
  """
  Encode image data to 12-bit packed format.

  Args:
      image: Input image tensor (uint16 or float32)
      format_type: "standard" or "ids" format

  Returns:
      Packed 12-bit data as uint8 tensor
  """
  ids = format_type is Packed12Format.IDS
  if image.dtype == torch.uint16:
    return extension.encode12_u16(image, ids_format=ids)
  if image.dtype == torch.float32:
    return extension.encode12_float(image, ids_format=ids)
  raise ValueError(f'Unsupported input dtype: {image.dtype}')


@beartype
def decode12(
  packed_data: torch.Tensor,
  output_dtype: torch.dtype = torch.float32,
  format_type: Packed12Format = Packed12Format.STANDARD,
) -> torch.Tensor:
  """
  Decode 12-bit packed data to image format.

  Args:
      packed_data: Packed 12-bit data as uint8 tensor
      output_dtype: torch.float32, torch.float16, or torch.uint16
      format_type: "standard" or "ids" format

  Returns:
      Decoded image tensor
  """
  ids = format_type is Packed12Format.IDS
  if output_dtype == torch.float32:
    return extension.decode12_float(packed_data, ids_format=ids)
  if output_dtype == torch.float16:
    return extension.decode12_half(packed_data, ids_format=ids)
  if output_dtype == torch.uint16:
    return extension.decode12_u16(packed_data, ids_format=ids)
  raise ValueError(f'Unsupported output dtype: {output_dtype}')


# Direct extension function access for individual functions
encode12_u16 = beartype(extension.encode12_u16)
encode12_float = beartype(extension.encode12_float)
decode12_float = beartype(extension.decode12_float)
decode12_half = beartype(extension.decode12_half)
decode12_u16 = beartype(extension.decode12_u16)


@beartype
def bilinear5x5_demosaic(image: torch.Tensor, bayer_pattern: BayerPattern) -> torch.Tensor:
  """
  Apply 5x5 bilinear demosaic to Bayer image.

  Args:
      image: Input Bayer image tensor (H, W, 1)
      bayer_pattern: Bayer pattern enumeration

  Returns:
      Demosaiced RGB image tensor (H, W, 3)
  """
  return extension.bilinear5x5_demosaic(image, bayer_pattern.value)


__all__ = [
  'BayerPattern',
  'Packed12Format',
  'bilinear5x5_demosaic',
  'create_postprocess',
  'create_ppg',
  'create_rcd',
  'decode12',
  'decode12_float',
  'decode12_half',
  'decode12_u16',
  'encode12',
  'encode12_float',
  'encode12_u16',
]
