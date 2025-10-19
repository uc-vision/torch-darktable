"""Image processing presets."""

from beartype import beartype

from .config import ImageProcessingSettings, ToneMapper


@beartype
def get_preset(name: str) -> ImageProcessingSettings:
  """Get a preset by name."""
  if name not in presets:
    raise ValueError(f'Unknown preset: {name}. Available: {list(presets.keys())}')
  return presets[name]


adaptive_aces = ImageProcessingSettings(
  enable_denoise=True,
  enable_bilateral=True,
  postprocess=True,
  tone_gamma=1.5,
  tone_intensity=2.0,
  light_adapt=0.8,
  tone_mapping=ToneMapper.adaptive_aces,
  vibrance=0.5,
)


aces = ImageProcessingSettings(
  enable_denoise=True,
  enable_bilateral=True,
  postprocess=True,
  tone_gamma=2.2,
  tone_intensity=1.0,
  tone_mapping=ToneMapper.aces,
  vibrance=0.5,
)

reinhard = ImageProcessingSettings(
  enable_denoise=True,
  enable_bilateral=True,
  postprocess=True,
  tone_gamma=1.0,
  tone_intensity=2.5,
  light_adapt=0.8,
  tone_mapping=ToneMapper.reinhard,
  vibrance=0.5,
)

presets: dict[str, ImageProcessingSettings] = {
  'aces': aces,
  'adaptive_aces': adaptive_aces,
  'reinhard': reinhard,
}
