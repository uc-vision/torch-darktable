from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from beartype import beartype


class ToneMapper(Enum):
  linear = 0
  reinhard = 1
  aces = 2
  adaptive_aces = 3


class Debayer(Enum):
  bilinear = 0
  ppg = 1
  rcd = 2


def clamp(x, lower, upper):
  return min(max(x, lower), upper)


@beartype
def float_field(default: float, range: tuple[float, float], description: str) -> Any:
  return field(default=default, metadata={'description': description, 'min': range[0], 'max': range[1]})


@beartype
def int_field(default: int, range: tuple[int, int], description: str, step: int | None = None) -> Any:
  return field(default=default, metadata={'description': description, 'min': range[0], 'max': range[1], 'step': step})


@beartype
def bool_field(default: bool, description: str) -> Any:
  return field(default=default, metadata={'description': description})


@beartype
def enum_field(default: Enum, description: str) -> Any:
  choices = [e.name for e in type(default)]
  return field(default=default, metadata={'description': description, 'choices': choices})


float3 = tuple[float, float, float]


@beartype
def float3_field(default: float3, range: tuple[float, float], description: str) -> Any:
  return field(default=default, metadata={'description': description, 'min': range[0], 'max': range[1]})


@beartype
@dataclass
class ImageProcessingSettings:
  # Tonemapping parameters
  tone_gamma: float = float_field(0.75, range=(0.1, 5.0), description='Gamma')
  tone_intensity: float = float_field(2.0, range=(-1.0, 5.0), description='Intensity')
  light_adapt: float = float_field(1.0, range=(0.0, 1.0), description='Light adaptation')

  # Vibrance adjustment
  vibrance: float = float_field(0.0, range=(-1.0, 1.0), description='Vibrance')

  # Temporal averaging to smooth intensity scaling over time (set to 1 to disable)
  moving_average: float = float_field(0.02, range=(0.0, 1.0), description='Tonemap moving average')

  debayer: Debayer = enum_field(Debayer.rcd, description='Debayer algorithm')
  ppg_median_threshold: float = 0.0

  postprocess: bool = bool_field(False, description='Postprocess debayer')
  green_eq_threshold: float = 0.04
  color_smoothing_passes: int = 3

  enable_bilateral: bool = bool_field(False, description='Enable bilateral constrast enhancement')
  bilateral: float = float_field(0.4, range=(0.0, 1.0), description='Bilateral constrast enhancement amount')

  bil_sigma_spatial: float = 2.0
  bil_sigma_luminance: float = 0.2

  enable_denoise: bool = bool_field(True, description='Enable denoise')
  denoise: float = float_field(0.075, range=(0.0, 1.0), description='Denoise amount')

  # linear or reinhard
  tone_mapping: ToneMapper = enum_field(ToneMapper.reinhard, description='Tonemapping algorithm')

  resize_width: int = int_field(0, range=(0, 4096), description='Resize width')
