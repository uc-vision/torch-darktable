from pathlib import Path
from typing import Annotated, Literal
import warnings

from beartype import beartype
from pydantic import BaseModel
import torch

import torch_darktable as td
from torch_darktable.pipeline.config import EnumValidator, ImageProcessingSettings
from torch_darktable.pipeline.transform import ImageTransform

warnings.filterwarnings('ignore', category=UserWarning, message='The given buffer is not writable')


@beartype
class CameraSettings(BaseModel, frozen=True):
  type: Literal['camera_settings'] = 'camera_settings'

  name: str
  image_size: tuple[int, int]
  padding: int = 0

  bayer_pattern: Annotated[td.BayerPattern, EnumValidator(td.BayerPattern, 'Bayer pattern')] = td.BayerPattern.RGGB
  packed_format: Annotated[td.PackedFormat, EnumValidator(td.PackedFormat, 'Packed format')] = td.PackedFormat.Packed12
  white_balance: tuple[float, float, float] | None = None
  image_processing: ImageProcessingSettings

  transform: Annotated[
    ImageTransform | dict[str, ImageTransform], EnumValidator(ImageTransform, 'Image transform')
  ] = ImageTransform.none

  def get_image_transform(self, camera_name: str) -> ImageTransform:
    if isinstance(self.transform, dict):
      return self.transform.get(camera_name, ImageTransform.none)
    return self.transform

  @property
  def bytes(self) -> int:
    return ((self.image_size[0] * self.image_size[1] * 3) // 2) + self.padding

  @beartype
  def save_json(self, path: Path) -> None:
    """Save settings to a JSON file."""
    path.write_text(self.model_dump_json(indent=2))

  @classmethod
  @beartype
  def load_json(cls, path: Path) -> 'CameraSettings':
    """Load settings from a JSON file."""
    return cls.model_validate_json(path.read_text())


@beartype
def load_raw_bytes(filepath: Path, device: torch.device = torch.device('cuda:0')):
  """Load raw image bytes into torch tensor without any decoding"""
  with filepath.open('rb') as f:
    raw_bytes = f.read()

  # disable user warning about writable buffer
  return torch.frombuffer(raw_bytes, dtype=torch.uint8).to(device, non_blocking=True)


@beartype
def load_raw_bytes_stripped(
  filepath: Path, camera_settings: CameraSettings, device: torch.device = torch.device('cuda:0')
):
  """Load raw image bytes and strip padding, but don't decode"""
  raw_bytes = load_raw_bytes(filepath, device)
  if camera_settings.padding > 0:
    raw_bytes = raw_bytes[: -camera_settings.padding]
  return raw_bytes


def load_raw_bayer(
  filepath: Path, camera_settings: CameraSettings | None = None, device: torch.device = torch.device('cuda:0')
) -> torch.Tensor:
  if camera_settings is None:
    camera_settings = settings_for_file(filepath)

  width, _height = camera_settings.image_size
  raw_cuda = load_raw_bytes(filepath, device).to(device, non_blocking=True)

  if camera_settings.padding > 0:
    raw_cuda = raw_cuda[: -camera_settings.padding]

  decoded = td.decode12(raw_cuda, output_dtype=torch.float32, format_type=camera_settings.packed_format)

  return decoded.view(-1, width)


def get_camera_settings_dir() -> Path:
  """Get the camera settings directory path."""
  return Path(__file__).parent.parent / 'camera_settings'


def load_camera_settings_from_dir(settings_dir: Path | None = None) -> dict[str, CameraSettings]:
  """Load all camera settings from JSON files in the settings directory."""
  if settings_dir is None:
    settings_dir = get_camera_settings_dir()

  settings = {}
  for json_file in settings_dir.glob('*.json'):
    camera_setting = CameraSettings.load_json(json_file)
    settings[camera_setting.name] = camera_setting

  return settings


def settings_for_file(file_path: Path) -> CameraSettings:
  """Get camera settings for a file.

  First tries directory name, then falls back to file size matching.
  """
  all_settings = load_camera_settings_from_dir()

  # Try directory name first
  camera_name = file_path.parent.stem
  if camera_name in all_settings:
    return all_settings[camera_name]

  # Fallback: match by file size
  file_size = file_path.stat().st_size
  for settings in all_settings.values():
    if settings.bytes == file_size:
      return settings

  raise ValueError(
    f'Could not find camera settings for "{file_path}". '
    f'Directory name "{camera_name}" not recognized and file size {file_size} bytes does not match any known camera. '
    f'Available cameras: {list(all_settings.keys())}'
  )


@beartype
def validate_camera_names(settings: CameraSettings, camera_names: list[str]) -> None:
  """Validate that camera names match what camera settings expects."""
  if isinstance(settings.transform, dict):
    expected_cameras = set(settings.transform.keys())
    actual_cameras = set(camera_names)
    if expected_cameras != actual_cameras:
      raise ValueError(
        f'Camera names mismatch: settings expects {sorted(expected_cameras)}, got {sorted(actual_cameras)}'
      )
