"""Bayer demosaicing algorithms and utilities."""

from beartype import beartype
import torch

from .bayer import BayerPattern, PackedFormat
from .extension import extension


class Bilinear5x5:
  def __init__(self, bayer_pattern: BayerPattern):
    self.bayer_pattern = bayer_pattern

  def process(self, image: torch.Tensor) -> torch.Tensor:
    return bilinear5x5_demosaic(image, self.bayer_pattern)


class PPG:
  """PPG demosaic with shape validation."""

  @beartype
  def __init__(
    self,
    device: torch.device,
    image_size: tuple[int, int],
    bayer_pattern: BayerPattern,
    *,
    median_threshold: float = 0.0,
  ):
    width, height = image_size
    self._ppg = extension.PPG(device, width, height, bayer_pattern.value, median_threshold)

  def process(self, input_tensor: torch.Tensor) -> torch.Tensor:
    expected_shape = (self._ppg.height, self._ppg.width, 1)
    if input_tensor.shape != expected_shape:
      raise RuntimeError(f'PPG input shape {input_tensor.shape} != expected {expected_shape}')
    return self._ppg.process(input_tensor)

  @property
  def image_size(self) -> tuple[int, int]:
    return (self._ppg.width, self._ppg.height)

  @property
  def median_threshold(self) -> float:
    return self._ppg.median_threshold


class RCD:
  """RCD demosaic with shape validation."""

  @beartype
  def __init__(
    self,
    device: torch.device,
    image_size: tuple[int, int],
    bayer_pattern: BayerPattern,
  ):
    width, height = image_size
    self._rcd = extension.RCD(device, width, height, bayer_pattern.value)

  def process(self, input_tensor: torch.Tensor) -> torch.Tensor:
    expected_shape = (self._rcd.height, self._rcd.width, 1)
    if input_tensor.shape != expected_shape:
      raise RuntimeError(f'RCD input shape {input_tensor.shape} != expected {expected_shape}')
    return self._rcd.process(input_tensor)

  @property
  def image_size(self) -> tuple[int, int]:
    return (self._rcd.width, self._rcd.height)


class PostProcess:
  """PostProcess with shape validation."""

  @beartype
  def __init__(
    self,
    device: torch.device,
    image_size: tuple[int, int],
    bayer_pattern: BayerPattern,
    *,
    color_smoothing_passes: int = 0,
    green_eq_local: bool = False,
    green_eq_global: bool = False,
    green_eq_threshold: float = 0.04,
  ):
    width, height = image_size
    self._postprocess = extension.PostProcess(
      device,
      width,
      height,
      bayer_pattern.value,
      color_smoothing_passes,
      green_eq_local,
      green_eq_global,
      green_eq_threshold,
    )

  def process(self, input_tensor: torch.Tensor) -> torch.Tensor:
    expected_shape = (self._postprocess.height, self._postprocess.width, 3)
    if input_tensor.shape != expected_shape:
      raise RuntimeError(f'PostProcess input shape {input_tensor.shape} != expected {expected_shape}')
    return self._postprocess.process(input_tensor)

  @property
  def image_size(self) -> tuple[int, int]:
    return (self._postprocess.width, self._postprocess.height)

  @property
  def color_smoothing_passes(self) -> int:
    return self._postprocess.color_smoothing_passes

  @property
  def green_eq_threshold(self) -> float:
    return self._postprocess.green_eq_threshold


# 12-bit packing/unpacking functions
@beartype
def encode(
  image: torch.Tensor, format_type: PackedFormat = PackedFormat.Packed12, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  """
  Encode image data to 12-bit packed format.

  Args:
      image: Input image tensor (uint16 or float32)
      format_type: "standard" or "ids" format

  Returns:
      Packed 12-bit data as uint8 tensor
  """
  assert dtype in {torch.float32, torch.uint16}

  ids = format_type is PackedFormat.Packed12_IDS
  if image.dtype == torch.uint16:
    return extension.encode12_u16(image, ids_format=ids)
  if image.dtype == torch.float32:
    return extension.encode12_float(image, ids_format=ids)
  raise ValueError(f'Unsupported input dtype: {image.dtype}')


@beartype
def decode12(
  packed_data: torch.Tensor,
  output_dtype: torch.dtype = torch.float32,
  format_type: PackedFormat = PackedFormat.Packed12,
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
  ids = format_type is PackedFormat.Packed12_IDS
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
  'PPG',
  'RCD',
  'BayerPattern',
  'Bilinear5x5',
  'PackedFormat',
  'PostProcess',
  'bilinear5x5_demosaic',
  'decode12',
  'decode12_float',
  'decode12_half',
  'decode12_u16',
  'encode',
  'encode12_float',
  'encode12_u16',
]
