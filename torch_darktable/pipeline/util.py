import torch


def lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
  return a + (b - a) * t


def lerp_none(a: torch.Tensor | None, b: torch.Tensor, t: float) -> torch.Tensor:
  if a is None:
    return b
  return lerp(a, b, t)


@torch.compile
def normalize_image(rgb_raw: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
  return (rgb_raw - bounds[0]) / (bounds[1] - bounds[0])


@torch.compile
def resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
  image = image.unsqueeze(0).permute(0, 3, 1, 2)
  image = torch.nn.functional.interpolate(image, size=size, mode='bilinear', align_corners=False)
  return image.permute(0, 2, 3, 1).squeeze(0).contiguous()


def resize_image(image: torch.Tensor, longest: int) -> torch.Tensor:
  h, w = image.shape[:2]
  size = resize_longest_edge((w, h), longest)

  return resize(image, size)


def resize_longest_edge(size: tuple[int, int], longest: int) -> tuple[int, int]:
  if longest == 0:
    return size

  if size[0] > size[1]:
    return (longest, size[1] * longest // size[0])

  return (size[0] * longest // size[1], longest)
