from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from beartype import beartype
import cv2
import numpy as np
import torch

import torch_darktable as td


class ImageTransform(Enum):
  none = 'none'
  rotate_90 = 'rotate_90'
  rotate_180 = 'rotate_180'
  rotate_270 = 'rotate_270'
  transpose = 'transpose'
  flip_horiz = 'flip_horiz'
  flip_vert = 'flip_vert'
  transverse = 'transverse'


@beartype
@dataclass
class CameraSettings:
  name: str
  image_size: tuple[int, int]
  ids_format: bool = False
  bayer_pattern: td.BayerPattern = td.BayerPattern.RGGB
  white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
  brightness: float = 1.0
  padding: int = 0
  transform: ImageTransform = ImageTransform.none
  preset: str = 'reinhard'

  @property
  def bytes(self) -> int:
    return (self.image_size[0] * self.image_size[1] * 3 // 2) + self.padding


@beartype
def transformed_size(original_size: tuple[int, int], transform: ImageTransform) -> tuple[int, int]:
  if transform in {ImageTransform.rotate_90, ImageTransform.rotate_270, ImageTransform.transpose}:
    return (original_size[1], original_size[0])  # swap width/height
  return original_size


def transform(image: torch.Tensor, transform: ImageTransform):
  if transform == ImageTransform.none:
    return image
  if transform == ImageTransform.rotate_90:
    return torch.rot90(image, 1, (0, 1)).contiguous()
  if transform == ImageTransform.rotate_180:
    return torch.rot90(image, 2, (0, 1)).contiguous()
  if transform == ImageTransform.rotate_270:
    return torch.rot90(image, 3, (0, 1)).contiguous()
  if transform == ImageTransform.flip_horiz:
    return torch.flip(image, (1,)).contiguous()
  if transform == ImageTransform.flip_vert:
    return torch.flip(image, (0,)).contiguous()
  if transform == ImageTransform.transverse:
    return torch.flip(image, (0, 1)).contiguous()
  if transform == ImageTransform.transpose:
    return torch.transpose(image, 0, 1).contiguous()
  return None


@beartype
def load_raw_bytes(filepath: Path, device: torch.device = torch.device('cuda')):
  """Load raw image bytes into torch tensor without any decoding"""
  with open(filepath, 'rb') as f:
    raw_bytes = f.read()
  return torch.frombuffer(raw_bytes, dtype=torch.uint8).to(device, non_blocking=True)


def load_raw_image(
  filepath: Path, camera_settings: CameraSettings | None = None, device: torch.device = torch.device('cuda')
) -> torch.Tensor:
  if camera_settings is None:
    camera_settings = settings_for_file(filepath)

  width, _height = camera_settings.image_size
  raw_cuda = load_raw_bytes(filepath, device).to(device, non_blocking=True)

  if camera_settings.padding > 0:
    raw_cuda = raw_cuda[: -camera_settings.padding]

  fmt = td.Packed12Format.IDS if camera_settings.ids_format else td.Packed12Format.STANDARD
  decoded = td.decode12(raw_cuda, output_dtype=torch.float32, format_type=fmt)

  bayer = decoded.view(-1, width)
  return scale_bayer(bayer, camera_settings.white_balance) * camera_settings.brightness


camera_settings = dict(
  blackfly=CameraSettings(
    name='blackfly',
    image_size=(4096, 3000),
    ids_format=False,
    white_balance=(1.0, 1.0, 1.0),
    brightness=0.8,
    transform=ImageTransform.rotate_270,
  ),
  ids=CameraSettings(
    name='ids',
    image_size=(2472, 2062),
    ids_format=True,
    white_balance=(1.5, 1.0, 1.5),
    brightness=1.0,
    transform=ImageTransform.rotate_90,
  ),
  pfr=CameraSettings(
    name='pfr',
    image_size=(4112, 3008),
    white_balance=(1.0, 1.0, 1.0),
    brightness=1.0,
    padding=1536,
    transform=ImageTransform.rotate_90,
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


def stack_bayer(bayer_image):
  return torch.stack(
    (
      bayer_image[0::2, 0::2],  # Red
      bayer_image[0::2, 1::2],  # Green
      bayer_image[1::2, 0::2],  # Green
      bayer_image[1::2, 1::2],  # Blue
    ),
    dim=-1,
  )


def expand_bayer(x):
  h, w = x.shape[0], x.shape[1]
  result = torch.zeros(h * 2, w * 2, device=x.device, dtype=x.dtype)

  r, g1, g2, b = x.unbind(dim=-1)

  result[0::2, 0::2] = r  # Red
  result[0::2, 1::2] = g1  # Green
  result[1::2, 0::2] = g2  # Green
  result[1::2, 1::2] = b  # Blue
  return result


def scale_bayer(x, white_balance=(0.5, 1.0, 0.5)):
  r, g, b = white_balance
  scaling = torch.tensor([r, g, g, b], device=x.device, dtype=x.dtype)

  x = stack_bayer(x) * scaling
  return expand_bayer(x)
