from enum import Enum

import torch

# Bayer pattern enumeration
class BayerPattern(Enum):
  RGGB: int
  BGGR: int
  GRBG: int
  GBRG: int

# Demosaicing classes
class PPG:
  def __init__(
    self,
    device: torch.device,
    width: int,
    height: int,
    pattern: int,
    median_threshold: float = ...,
  ) -> None: ...
  def process(self, input: torch.Tensor) -> torch.Tensor: ...
  median_threshold: float

class RCD:
  def __init__(
    self,
    device: torch.device,
    width: int,
    height: int,
    pattern: int,
  ) -> None: ...
  def process(self, input: torch.Tensor) -> torch.Tensor: ...

class PostProcess:
  def __init__(
    self,
    device: torch.device,
    width: int,
    height: int,
    pattern: int,
    color_smoothing_passes: int = ...,
    green_eq_local: bool = ...,
    green_eq_global: bool = ...,
    green_eq_threshold: float = ...,
  ) -> None: ...
  def process(self, input: torch.Tensor) -> torch.Tensor: ...
  color_smoothing_passes: int
  green_eq_local: bool
  green_eq_global: bool
  green_eq_threshold: float

# 12-bit encoding/decoding functions (internal - use high-level API instead)
def encode12_u16(image: torch.Tensor, ids_format: bool = ...) -> torch.Tensor: ...
def encode12_float(image: torch.Tensor, ids_format: bool = ..., scaled: bool = ...) -> torch.Tensor: ...
def decode12_float(packed_data: torch.Tensor, ids_format: bool = ..., scaled: bool = ...) -> torch.Tensor: ...
def decode12_half(packed_data: torch.Tensor, ids_format: bool = ..., scaled: bool = ...) -> torch.Tensor: ...
def decode12_u16(packed_data: torch.Tensor, ids_format: bool = ...) -> torch.Tensor: ...

# Local contrast classes
class Laplacian:
  def __init__(
    self,
    device: torch.device,
    width: int,
    height: int,
    num_gamma: int = ...,
    sigma: float = ...,
    shadows: float = ...,
    highlights: float = ...,
    clarity: float = ...,
  ) -> None: ...
  def process(self, input: torch.Tensor) -> torch.Tensor: ...
  sigma: float
  shadows: float
  highlights: float
  clarity: float

class Bilateral:
  def __init__(
    self,
    device: torch.device,
    width: int,
    height: int,
    sigma_s: float = ...,
    sigma_r: float = ...,
  ) -> None: ...
  def process(self, luminance: torch.Tensor, detail: float) -> torch.Tensor: ...
  sigma_s: float
  sigma_r: float

# Tone mapping parameters struct
class TonemapParams:
  def __init__(self, gamma: float, intensity: float, light_adapt: float) -> None: ...
  gamma: float
  intensity: float
  light_adapt: float

# Tone mapping functions
def compute_image_metrics(
  images: list[torch.Tensor], stride: int = ..., min_gray: float = ..., rescale: bool = ...
) -> torch.Tensor: ...
def compute_image_bounds(images: list[torch.Tensor], stride: int) -> torch.Tensor: ...

# Struct-based tonemap functions (C++ interface)
def reinhard_tonemap(image: torch.Tensor, metrics: torch.Tensor, params: TonemapParams) -> torch.Tensor: ...
def aces_tonemap(image: torch.Tensor, metrics: torch.Tensor, params: TonemapParams) -> torch.Tensor: ...
def linear_tonemap(image: torch.Tensor, metrics: torch.Tensor, params: TonemapParams) -> torch.Tensor: ...

# Color conversion functions
def compute_luminance(rgb: torch.Tensor) -> torch.Tensor: ...
def modify_luminance(rgb: torch.Tensor, new_luminance: torch.Tensor) -> torch.Tensor: ...
def compute_log_luminance(rgb: torch.Tensor, eps: float) -> torch.Tensor: ...
def modify_log_luminance(rgb: torch.Tensor, log_luminance: torch.Tensor, eps: float) -> torch.Tensor: ...
def modify_hsl(
  rgb: torch.Tensor,
  hue_adjust: float = 0.0,
  sat_adjust: float = 0.0,
  lum_adjust: float = 0.0,
) -> torch.Tensor: ...
def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor: ...
def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor: ...
def lab_to_xyz(lab: torch.Tensor) -> torch.Tensor: ...
def xyz_to_rgb(xyz: torch.Tensor) -> torch.Tensor: ...
def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor: ...
def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor: ...
def color_transform_3x3(input: torch.Tensor, matrix_3x3: torch.Tensor) -> torch.Tensor: ...

# Bilinear 5x5 demosaic function
def bilinear5x5_demosaic(input: torch.Tensor, pattern: int) -> torch.Tensor: ...

# White balance functions
def apply_white_balance(bayer_image: torch.Tensor, gains: torch.Tensor, pattern: int) -> torch.Tensor: ...
def estimate_white_balance(
  bayer_images: list[torch.Tensor],
  pattern: int,
  quantile: float = ...,
  stride: int = ...,
) -> torch.Tensor: ...

# Wiener denoising
class Wiener:
  def __init__(
    self,
    device: torch.device,
    width: int,
    height: int,
    overlap_factor: int = ...,
    tile_size: int = ...,
  ) -> None: ...
  def process(self, input: torch.Tensor, noise_sigmas: torch.Tensor) -> torch.Tensor: ...
  @property
  def overlap_factor(self) -> int: ...

# Wiener denoiser creation function
def create_wiener(
  device: torch.device,
  width: int,
  height: int,
  overlap_factor: int = ...,
  tile_size: int = ...,
) -> Wiener: ...
