from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, get_args, get_origin

from beartype import beartype
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema


class Validator:
  """Base class for all field validators."""
  description: str


class Float(Validator):
  def __init__(self, range: tuple[float, float], description: str):
    self.range = range
    self.description = description

  def __get_pydantic_core_schema__(self, _source_type, _handler: GetCoreSchemaHandler):
    def validate(v: float):
      v = float(v)
      if not (self.range[0] <= v <= self.range[1]):
        raise ValueError(f"{v} not in [{self.range[0]}, {self.range[1]}]")
      return v
    return core_schema.no_info_plain_validator_function(validate)


class Int(Validator):
  def __init__(self, range: tuple[int, int], description: str, step: int | None = None):
    self.range = range
    self.description = description
    self.step = step

  def __get_pydantic_core_schema__(self, _source_type, _handler: GetCoreSchemaHandler):
    def validate(v: int):
      v = int(v)
      if not (self.range[0] <= v <= self.range[1]):
        raise ValueError(f"{v} not in [{self.range[0]}, {self.range[1]}]")
      return v
    return core_schema.no_info_plain_validator_function(validate)


class Bool(Validator):
  def __init__(self, description: str):
    self.description = description

  def __get_pydantic_core_schema__(self, _source_type, _handler: GetCoreSchemaHandler):
    def validate(v: bool):
      return bool(v)
    return core_schema.no_info_plain_validator_function(validate)


class EnumValidator[TEnum: Enum](Validator):
  def __init__(self, enum_type: type[TEnum], description: str):
    self.enum_type = enum_type
    self.description = description

  def __get_pydantic_core_schema__(self, _source_type, _handler: GetCoreSchemaHandler):
    def validate(v):
      if isinstance(v, self.enum_type):
        return v
      if isinstance(v, str):
        return self.enum_type[v]
      if isinstance(v, dict):
        return {k: self.enum_type[val] if isinstance(val, str) else val for k, val in v.items()}
      raise ValueError(f"{v} is not a {self.enum_type.__name__}")

    def serialize(v):
      if isinstance(v, dict):
        return {k: val.name for k, val in v.items()}
      return v.name

    return core_schema.no_info_plain_validator_function(
      validate,
      serialization=core_schema.plain_serializer_function_ser_schema(serialize, when_used='always')
    )


def get_validator(model: type[BaseModel], field_name: str) -> Validator | None:
  """Extract the validator instance from a field's annotation."""
  annotation = model.__annotations__.get(field_name)
  if annotation is None:
    return None
  if get_origin(annotation) is Annotated:
    args = get_args(annotation)
    for arg in args[1:]:  # Skip the first arg (the actual type)
      if isinstance(arg, Validator):
        return arg
  return None


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


class ImageProcessingSettings(BaseModel, frozen=True):
  type: Literal['image_processing_settings'] = 'image_processing_settings'

  # Tonemapping parameters
  tone_gamma: Annotated[float, Float(range=(0.1, 5.0), description='Gamma')] = 0.75
  tone_intensity: Annotated[float, Float(range=(-1.0, 5.0), description='Intensity')] = 2.0
  light_adapt: Annotated[float, Float(range=(0.0, 1.0), description='Light adaptation')] = 1.0

  vibrance: Annotated[float, Float(range=(-1.0, 1.0), description='Vibrance')] = 0.0

  # Temporal averaging to smooth intensity scaling over time (set to 1 to disable)
  moving_average: Annotated[float, Float(range=(0.0, 1.0), description='Tonemap moving average')] = 0.02

  debayer: Annotated[Debayer, EnumValidator(Debayer, description='Debayer algorithm')] = Debayer.rcd
  ppg_median_threshold: float = 0.0

  postprocess: Annotated[bool, Bool(description='Postprocess debayer')] = False
  green_eq_threshold: float = 0.04
  color_smoothing_passes: int = 3

  enable_bilateral: Annotated[bool, Bool(description='Enable bilateral constrast enhancement')] = False
  bilateral: Annotated[
    float, Float(range=(0.0, 1.0), description='Bilateral constrast enhancement amount')
  ] = 0.4

  bil_sigma_spatial: float = 2.0
  bil_sigma_luminance: float = 0.2

  enable_denoise: Annotated[bool, Bool(description='Enable denoise')] = True
  denoise: Annotated[float, Float(range=(0.0, 1.0), description='Denoise amount')] = 0.075

  # linear or reinhard
  tone_mapping: Annotated[
    ToneMapper, EnumValidator(ToneMapper, description='Tonemapping algorithm')
  ] = ToneMapper.reinhard

  resize_width: Annotated[int, Int(range=(0, 4096), description='Resize width')] = 0

  @beartype
  def save_json(self, path: Path) -> None:
    """Save settings to a JSON file."""
    path.write_text(self.model_dump_json(indent=2))

  @classmethod
  @beartype
  def load_json(cls, path: Path) -> 'ImageProcessingSettings':
    """Load settings from a JSON file."""
    return cls.model_validate_json(path.read_text())
