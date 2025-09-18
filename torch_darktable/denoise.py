from beartype import beartype
from .extension import extension
import torch


def check_overlap_factor(overlap_factor: int):
  if overlap_factor not in [2, 4, 8]:
    raise ValueError('overlap_factor must be 2, 4, or 8')


class Wiener:
  """High-level Wiener denoiser with flexible noise parameter handling."""

  @beartype
  def __init__(
    self,
    device: torch.device,
    image_size: tuple[int, int],
    overlap_factor: int = 4,
    tile_size: int = 32,
  ):
    width, height = image_size

    # Validate parameters
    if not device.type == 'cuda':
      raise ValueError(f'Device must be CUDA, got {device}')
    if width <= 0 or height <= 0:
      raise ValueError(f'Image dimensions must be positive, got {width}x{height}')
    check_overlap_factor(overlap_factor)
    if tile_size not in [16, 32]:
      raise ValueError(f'tile_size must be 16 or 32, got {tile_size}')

    try:
      self._wiener = extension.Wiener(device, width, height, overlap_factor, tile_size)
    except Exception as e:
      raise RuntimeError(f'Failed to create Wiener extension: {e}') from e

    self._tile_size = tile_size
    self._device = device

  def __repr__(self):
    return f'Wiener(overlap_factor={self.overlap_factor} tile_size={self._tile_size})'

  def process_luminance(self, image: torch.Tensor, noise: float | torch.Tensor) -> torch.Tensor:
    # Process luminance channel only (single channel processing)
    luminance = extension.compute_luminance(image)
    modified = self.process(luminance.unsqueeze(2), noise).squeeze(2)
    return extension.modify_luminance(image, modified)

  def process_log_luminance(self, image: torch.Tensor, noise: float | torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    log_luminance = extension.compute_log_luminance(image, eps=eps)
    modified = self.process(log_luminance.unsqueeze(2), noise).squeeze(2)
    return extension.modify_log_luminance(image, modified, eps=eps)

  def process_log(self, image: torch.Tensor, noise: float | torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    log_rgb = (image + eps).log()
    return self.process(log_rgb, noise).exp()

  @property
  def overlap_factor(self) -> int:
    return self._wiener.overlap_factor

  @beartype
  def process(self, image: torch.Tensor, noise: float | torch.Tensor) -> torch.Tensor:
    """
    Process image with Wiener filter.

    Args:
        image: RGB image tensor of shape (H, W, channels)
        noise: Noise levels
          - float (all channels),
          - Tensor[channels] (per-channel)

    Returns:
        Denoised image of same shape
    """
    # Determine channels from input image
    channels = image.size(2)
    if channels not in [1, 3]:
      raise ValueError(f'image channels must be 1 or 3, got {channels}')

    if isinstance(noise, float):
      # Single float for all channels
      noise_sigmas = torch.tensor([noise] * channels, dtype=torch.float32, device=self._device)
    elif isinstance(noise, torch.Tensor):
      if noise.shape != (channels,):
        raise ValueError(f'noise tensor must have {channels} elements for {channels}-channel image')
      noise_sigmas = noise.to(dtype=torch.float32, device=self._device)
    else:
      raise ValueError(f'noise must be float, or Tensor[{channels}]')

    return self._wiener.process(image, noise_sigmas)


@beartype
def create_wiener(
  device: torch.device,
  image_size: tuple[int, int],
  *,
  overlap: int = 4,
  tile_size: int = 32,
) -> Wiener:
  """
  Create a Wiener denoiser object with flexible noise handling.

  Args:
      device: CUDA device to use
      image_size: (width, height) of the image
      overlap: Overlap factor (2=half, 4=quarter, 8=eighth block overlap)
      tile_size: Tile size (16 for 16x16 tiles, 32 for 32x32 tiles)

  Returns:
      High-level Wiener denoiser object that can handle both 1 and 3 channel images
  """
  return Wiener(device, image_size, overlap_factor=overlap, tile_size=tile_size)


@beartype
def estimate_channel_noise(image: torch.Tensor, stride: int = 8) -> torch.Tensor:
  """
  Estimate per-channel noise levels from image using high-frequency analysis.

  Based on paper's MAD approach: σ ≈ median(|H - median(H)|) / 0.6745

  Args:
      image: RGB image tensor of shape (H, W, 3)
      stride: Subsampling stride for faster computation (default: 8)

  Returns:
      Per-channel noise sigmas as tensor of shape (3,) for R, G, B channels
  """
  # Laplacian kernel for high-frequency extraction
  laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=image.dtype, device=image.device)

  # Apply to all channels at once: (H, W, 3) → (1, 3, H, W) for conv2d
  image_chw = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
  laplacian_3ch = laplacian.unsqueeze(0).repeat(3, 1, 1).unsqueeze(1)  # (3, 1, 3, 3)

  # 3-channel convolution (one kernel per channel)
  high_freq = torch.conv2d(image_chw, laplacian_3ch, groups=3, padding=1)  # (1, 3, H, W)

  # Subsample with stride for speed - much faster on large images
  high_freq_subsampled = high_freq[0, :, ::stride, ::stride]  # (channels, H//stride, W//stride)
  high_freq_flat = high_freq_subsampled.flatten(1)  # (channels, subsampled_pixels)

  # Compute MAD per channel using proper median approach
  median_vals = torch.median(high_freq_flat, dim=1).values  # (channels,)
  mad = torch.median(torch.abs(high_freq_flat - median_vals.unsqueeze(1)), dim=1).values  # (channels,)
  return mad / 0.6745
