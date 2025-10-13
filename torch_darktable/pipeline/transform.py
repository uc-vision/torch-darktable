from enum import Enum

from beartype import beartype
import torch


class ImageTransform(Enum):
  none = 0
  rotate_90 = 1
  rotate_180 = 2
  rotate_270 = 3
  transpose = 4
  flip_horiz = 5
  flip_vert = 6
  transverse = 7

  def next_rotation(self) -> 'ImageTransform':
    """Return the next rotation in 90-degree increments."""
    rotation_map = {
      ImageTransform.none: ImageTransform.rotate_90,
      ImageTransform.rotate_90: ImageTransform.rotate_180,
      ImageTransform.rotate_180: ImageTransform.rotate_270,
      ImageTransform.rotate_270: ImageTransform.none,
      ImageTransform.transpose: ImageTransform.flip_horiz,
      ImageTransform.flip_horiz: ImageTransform.flip_vert,
      ImageTransform.flip_vert: ImageTransform.transverse,
      ImageTransform.transverse: ImageTransform.transpose,
    }
    return rotation_map.get(self, ImageTransform.rotate_90)


@beartype
def transformed_size(original_size: tuple[int, int], transform: ImageTransform) -> tuple[int, int]:
  if transform in {ImageTransform.rotate_90, ImageTransform.rotate_270, ImageTransform.transpose}:
    return (original_size[1], original_size[0])  # swap width/height
  return original_size


def transform(image: torch.Tensor, transform: ImageTransform):  # noqa: PLR0911
  match transform:
    case ImageTransform.none:
      return image
    case ImageTransform.rotate_90:
      return torch.rot90(image, 1, (0, 1)).contiguous()
    case ImageTransform.rotate_180:
      return torch.rot90(image, 2, (0, 1)).contiguous()
    case ImageTransform.rotate_270:
      return torch.rot90(image, 3, (0, 1)).contiguous()
    case ImageTransform.flip_horiz:
      return torch.flip(image, (1,)).contiguous()
    case ImageTransform.flip_vert:
      return torch.flip(image, (0,)).contiguous()
    case ImageTransform.transverse:
      return torch.flip(image, (0, 1)).contiguous()
    case ImageTransform.transpose:
      return torch.transpose(image, 0, 1).contiguous()
