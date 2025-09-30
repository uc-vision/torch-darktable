from enum import Enum
from pathlib import Path

from beartype import beartype
import cv2
import numpy as np
import torch

from .extension import extension


class BayerPattern(Enum):
  RGGB = extension.BayerPattern.RGGB
  BGGR = extension.BayerPattern.BGGR
  GRBG = extension.BayerPattern.GRBG
  GBRG = extension.BayerPattern.GBRG


class PackedFormat(Enum):
  Packed12 = 0
  Packed12_IDS = 1



@beartype
def rgb_to_bayer(rgb_tensor: torch.Tensor, pattern: BayerPattern = BayerPattern.RGGB) -> torch.Tensor:
  """Convert RGB tensor to Bayer pattern.

  Args:
      rgb_tensor: RGB tensor of shape (H, W, 3)
      pattern: Bayer pattern

  Returns:
      Bayer tensor of shape (H, W, 1)
  """
  c1, c2, c3, c4 = channels(pattern)

  stacked = torch.stack((
    rgb_tensor[0::2, 0::2, c1],  # R (even rows, even cols)
    rgb_tensor[0::2, 1::2, c2],  # G (even rows, odd cols)
    rgb_tensor[1::2, 0::2, c3],  # G (odd rows, even cols)
    rgb_tensor[1::2, 1::2, c4],  # B (odd rows, odd cols)
  ), dim=-1)

  return expand_bayer(stacked)


@beartype
def load_as_bayer(image_path: Path,
  pattern: BayerPattern = BayerPattern.RGGB,
  device: torch.device = torch.device('cuda')
  ) -> torch.Tensor:
  """Load RGB image as torch tensor on GPU.

  Returns:
      RGB tensor of shape (H, W, 3) with values in [0, 1], on CUDA
  """
  if not image_path.exists():
    raise FileNotFoundError(f'Image not found: {image_path}')

  image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  image = image.astype(np.float32) / 255.0
  image = torch.from_numpy(image).to(device)
  return rgb_to_bayer(image, pattern)


def pixel_order(pattern: BayerPattern) -> tuple[int, int, int, int]:
  match pattern:
    case BayerPattern.RGGB:
      return (0, 1, 2, 3)
    case BayerPattern.BGGR:
      return (3, 1, 2, 0)
    case BayerPattern.GRBG:
      return (1, 0, 3, 2)
    case BayerPattern.GBRG:
      return (1, 3, 0, 2)

  raise ValueError(f'Invalid bayer pattern: {pattern}')


def channels(pattern: BayerPattern) -> tuple[int, int, int, int]:
  match pattern:
    case BayerPattern.RGGB:
      return (0, 1, 1, 2)
    case BayerPattern.BGGR:
      return (2, 1, 1 ,0)
    case BayerPattern.GRBG:
      return (1, 0, 1, 2)
    case BayerPattern.GBRG:
      return (1, 2, 1, 0)

  raise ValueError(f'Invalid bayer pattern: {pattern}')


def stack_bayer(bayer_image: torch.Tensor) -> torch.Tensor:
  return torch.stack((
    bayer_image[0::2, 0::2],   # Red
    bayer_image[0::2, 1::2],   # Green
    bayer_image[1::2, 0::2],   # Green
    bayer_image[1::2, 1::2]    # Blue
  ), dim=-1)


def expand_bayer(x):
    h, w = x.shape[0], x.shape[1]
    result = torch.zeros(h * 2, w * 2, device=x.device, dtype=x.dtype)

    r, g1, g2, b = x.unbind(dim=-1)

    result[0::2, 0::2] = r    # Red
    result[0::2, 1::2] = g1   # Green
    result[1::2, 0::2] = g2   # Green
    result[1::2, 1::2] = b    # Blue
    return result
