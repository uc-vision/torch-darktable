from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
import cv2
import numpy as np
import torch

import torch_darktable as td
from torch_darktable.pipeline.transform import ImageTransform


@beartype
@dataclass(frozen=True, kw_only=True)
class CameraSettings:
  name: str
  image_size: tuple[int, int]
  bayer_pattern: td.BayerPattern = td.BayerPattern.RGGB
  packed_format: td.PackedFormat = td.PackedFormat.Packed12
  padding: int = 0
  white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
  preset: str

  transform: ImageTransform | dict[str, ImageTransform] = ImageTransform.none

  def get_image_transform(self, camera_name: str) -> ImageTransform:
    if isinstance(self.transform, dict):
      return self.transform.get(camera_name, ImageTransform.none)
    return self.transform



  @property
  def bytes(self) -> int:
    return ((self.image_size[0] * self.image_size[1] * 3) // 2) + self.padding


@beartype
def load_raw_bytes(filepath: Path, device: torch.device = torch.device('cuda:0')):
  """Load raw image bytes into torch tensor without any decoding"""
  with filepath.open('rb') as f:
    raw_bytes = f.read()
  return torch.frombuffer(raw_bytes, dtype=torch.uint8).to(device, non_blocking=True)


@beartype
def load_raw_bytes_stripped(filepath: Path, camera_settings: CameraSettings, device: torch.device = torch.device('cuda:0')):
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


beetroot_transforms = {f"cam{i}": ImageTransform.rotate_270 if i > 6 else ImageTransform.rotate_90 for i in range(1, 13)}

camera_settings = dict(
  artichoke=CameraSettings(
    name='artichoke',
    image_size=(4096, 3000),
    packed_format=td.PackedFormat.Packed12,
    preset='adaptive_aces',
    transform=ImageTransform.rotate_270,
  ),

  # carrot=CameraSettings(
  #   name='carrot',
  #   image_size=(2472, 2062),
  #   packed_format=td.PackedFormat.Packed12_IDS,
  #   preset='adaptive_aces',
  #   transform=ImageTransform.rotate_270,
  #   white_balance=(1.8, 1.0, 2.1),
  # ),

  beetroot=CameraSettings(
    name='beetroot',
    image_size=(2472, 2062),
    packed_format=td.PackedFormat.Packed12_IDS,
    preset='adaptive_aces',
    transform=beetroot_transforms,
    white_balance=(1.8, 1.0, 2.1),
  ),

  pfr=CameraSettings(
    name='pfr',
    image_size=(4112, 3008),
    padding=1536,
    preset='adaptive_aces',
    packed_format=td.PackedFormat.Packed12,
    transform=ImageTransform.rotate_270,
  ),
)


def settings_for_file(file_path: Path) -> CameraSettings:
  global camera_settings

  camera_sizes = {camera_settings.name: camera_settings.bytes for camera_settings in camera_settings.values()}
  for name, size in camera_sizes.items():
    if size == file_path.stat().st_size:
      return camera_settings[name]

  raise ValueError(f'Could not match size of {file_path} with known camera settings {camera_sizes}')


def display_rgb(k: str, rgb_image: torch.Tensor | np.ndarray):
  if isinstance(rgb_image, torch.Tensor):
    rgb_image = rgb_image.cpu().numpy()
  cv2.namedWindow(k, cv2.WINDOW_NORMAL)

  # loop while wiow is not closed
  cv2.imshow(k, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
  while cv2.waitKey(1) & 255 != ord('q') or cv2.getWindowProperty(k, cv2.WND_PROP_VISIBLE) >= 1:
    pass

  cv2.destroyAllWindows()
