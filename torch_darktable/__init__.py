"""PyTorch CUDA extensions for image processing and computer vision."""

# Import all modules
from . import (
  color_conversion,
  debayer,
  denoise,
  extension,
  local_contrast,
  tonemap,
  white_balance,
)
from .color_conversion import (
  color_transform_3x3,
  compute_log_luminance,
  compute_luminance,
  lab_to_rgb,
  lab_to_xyz,
  modify_hsl,
  modify_log_luminance,
  modify_luminance,
  modify_vibrance,
  rgb_to_lab,
  rgb_to_xyz,
  xyz_to_lab,
  xyz_to_rgb,
)
from .debayer import (
  BayerPattern,
  PackedFormat,
  bilinear5x5_demosaic,
  create_postprocess,
  create_ppg,
  create_rcd,
  decode12,
  decode12_float,
  decode12_half,
  decode12_u16,
  encode,
  encode12_float,
  encode12_u16,
)
from .denoise import Wiener, create_wiener, estimate_channel_noise
from .local_contrast import (
  LaplacianParams,
  bilateral_rgb,
  create_bilateral,
  create_laplacian,
  local_laplacian_rgb,
)
from .tonemap import (
  TonemapParameters,
  aces_tonemap,
  compute_image_bounds,
  compute_image_metrics,
  print_metrics,
  metrics_from_dict,
  metrics_to_dict,

  linear_tonemap,
  reinhard_tonemap,
)
from .white_balance import apply_white_balance, estimate_white_balance

__all__ = [
  # Core classes and enums
  'BayerPattern',
  'LaplacianParams',
  'PackedFormat',
  'TonemapParameters',
  # Wiener denoising
  'Wiener',
  # Tone mapping
  'aces_tonemap',
  # White balance
  'apply_white_balance',
  'bilateral_rgb',
  'bilinear5x5_demosaic',
  'color_conversion',
  'color_transform_3x3',
  'compute_image_bounds',
  'compute_image_metrics',
  'compute_log_luminance',
  'metrics_from_dict',
  'metrics_to_dict',
  'print_metrics',
  # Color conversions
  'compute_luminance',
  'create_bilateral',
  # Local contrast enhancement
  'create_laplacian',
  'create_postprocess',
  # Debayering and 12-bit encoding
  'create_ppg',
  'create_rcd',
  'create_wiener',
  'debayer',
  'decode12',
  'decode12_float',
  'decode12_half',
  'decode12_u16',
  'denoise',
  'encode',
  'encode12_float',
  'encode12_u16',
  'estimate_channel_noise',
  'estimate_white_balance',
  # Submodules
  'extension',
  'lab_to_rgb',
  'lab_to_xyz',
  'linear_tonemap',
  'local_contrast',
  'local_laplacian_rgb',
  'modify_hsl',
  'modify_log_luminance',
  'modify_luminance',
  'modify_vibrance',
  'reinhard_tonemap',
  'rgb_to_lab',
  'rgb_to_xyz',
  'tonemap',
  'white_balance',
  'xyz_to_lab',
  'xyz_to_rgb',
]
