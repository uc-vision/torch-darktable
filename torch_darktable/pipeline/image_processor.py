"""Image processing pipeline with configurable settings and presets."""

from beartype import beartype
import torch

import torch_darktable as td
from torch_darktable.pipeline.camera_settings import CameraSettings
from torch_darktable.pipeline.transform import ImageTransform, transform

from .config import Debayer, ImageProcessingSettings, ToneMapper
from .util import lerp, normalize_image, resize_longest_edge


class ImageSizeMismatchError(Exception):
  """Raised when image size does not match expected dimensions."""

  def __init__(
    self,
    message: str,
    image_size: tuple[int, int],
    packed_format: td.PackedFormat,
    padding: int,
  ):
    super().__init__(message)
    self.image_size = image_size
    self.packed_format = packed_format
    self.padding = padding


@beartype
class ImageProcessor:
  @beartype
  def __init__(
    self,
    image_size: tuple[int, int],
    bayer_pattern: td.BayerPattern,
    packed_format: td.PackedFormat,
    settings: ImageProcessingSettings,
    device: torch.device,
    white_balance: tuple[float, float, float] | None,
    transforms: ImageTransform | dict[str, ImageTransform] = ImageTransform.none,
    padding: int = 0,
  ):
    """Initialize the pipeline with device, camera settings and processing settings.

    Args:
        image_size: Image dimensions as (width, height)
        bayer_pattern: Bayer pattern configuration
        packed_format: Raw data packing format
        settings: Processing settings
        device: CUDA device for processing
        white_balance: White balance coefficients from camera settings (None to disable)
        transforms: Image transforms (either single or per-camera)
        padding: Number of bytes to skip from the beginning of the image
    """

    assert device.index is not None, f'Device not fully specified: {device}'

    self.device = device
    self.settings = settings
    self.image_size = image_size
    self.bayer_pattern = bayer_pattern
    self.packed_format = packed_format
    self.transforms = transforms
    self.padding = padding

    self.metrics: torch.Tensor | None = None
    self.bounds: torch.Tensor | None = None

    self.bil_workspace = td.Bilateral(
      self.device, self.image_size, sigma_s=settings.bil_sigma_spatial, sigma_r=settings.bil_sigma_luminance
    )

    self.rcd_workspace = td.RCD(self.device, self.image_size, self.bayer_pattern)

    self.ppg_workspace = td.PPG(
      self.device, self.image_size, self.bayer_pattern, median_threshold=settings.ppg_median_threshold
    )

    self.postprocess_workspace = td.PostProcess(
      self.device,
      self.image_size,
      self.bayer_pattern,
      color_smoothing_passes=settings.color_smoothing_passes,
      green_eq_local=False,
      green_eq_global=True,
      green_eq_threshold=settings.green_eq_threshold,
    )

    self.wiener_workspace = td.Wiener(self.device, self.image_size)
    self.white_balance = (
      torch.tensor(white_balance, device=self.device).to(torch.float32) if white_balance is not None else None
    )

  def __repr__(self) -> str:
    wb_str = (
      f'({self.white_balance[0]:.3f}, {self.white_balance[1]:.3f}, {self.white_balance[2]:.3f})'
      if self.white_balance is not None
      else 'None'
    )
    transform_str = (
      f'{self.transforms.name}'
      if isinstance(self.transforms, ImageTransform)
      else f'{{{", ".join(f"{k}: {v.name}" for k, v in self.transforms.items())}}}'
    )
    return (
      f'ImageProcessor('
      f'size={self.image_size}, '
      f'bayer={self.bayer_pattern.name}, '
      f'format={self.packed_format.name}, '
      f'device={self.device}, '
      f'wb={wb_str}, '
      f'padding={self.padding}, '
      f'transform={transform_str}, '
      f'debayer={self.settings.debayer.name}, '
      f'tonemap={self.settings.tone_mapping.name})'
    )

  @staticmethod
  def from_camera_settings(camera_settings: CameraSettings, device: torch.device):
    image_settings = camera_settings.image_processing

    return ImageProcessor(
      camera_settings.image_size,
      camera_settings.bayer_pattern,
      camera_settings.packed_format,
      image_settings,
      device=device,
      white_balance=camera_settings.white_balance,
      transforms=camera_settings.transform,
      padding=camera_settings.padding,
    )

  def update_settings(self, settings: ImageProcessingSettings):
    old_settings = self.settings
    self.settings = settings

    def changed(*attrs: str) -> bool:
      return any(getattr(old_settings, attr) != getattr(settings, attr) for attr in attrs)

    if changed('bil_sigma_spatial', 'enable_bilateral', 'bil_sigma_luminance'):
      self.bil_workspace = td.Bilateral(
        self.device, self.image_size, sigma_s=settings.bil_sigma_spatial, sigma_r=settings.bil_sigma_luminance
      )

    if changed('ppg_median_threshold'):
      self.ppg_workspace = td.PPG(
        self.device, self.image_size, self.bayer_pattern, median_threshold=settings.ppg_median_threshold
      )

    if changed('color_smoothing_passes', 'green_eq_threshold'):
      self.postprocess_workspace = td.PostProcess(
        self.device,
        self.image_size,
        self.bayer_pattern,
        color_smoothing_passes=settings.color_smoothing_passes,
        green_eq_local=False,
        green_eq_global=True,
        green_eq_threshold=settings.green_eq_threshold,
      )

  @property
  def final_size(self):
    return resize_longest_edge(self.image_size, self.settings.resize_width)

  @property
  def expected_bytes(self) -> int:
    """Compute expected raw file size in bytes for this processor's configuration."""
    width, height = self.image_size
    pixels = width * height

    match self.packed_format:
      case td.PackedFormat.Packed12 | td.PackedFormat.Packed12_IDS:
        # 12-bit packed: 3 bytes per 2 pixels
        raw_bytes = (pixels * 3) // 2
      case _:
        raise ValueError(f'Unsupported packed format: {self.packed_format}')

    return raw_bytes + self.padding

  def _image_size_mismatch_error(self, message: str) -> ImageSizeMismatchError:
    return ImageSizeMismatchError(
      message,
      image_size=self.image_size,
      packed_format=self.packed_format,
      padding=self.padding,
    )

  @beartype
  def load_bytes(self, bytes: torch.Tensor) -> torch.Tensor:
    if bytes.numel() != self.expected_bytes:
      raise self._image_size_mismatch_error(
        f'Image size mismatch: expected {self.expected_bytes} bytes for {self.image_size} {self.packed_format.name} '
        f'with {self.padding} padding, got {bytes.numel()} bytes. '
      )

    if self.padding > 0:
      bytes = bytes[: -self.padding]

    conversion = {
      td.PackedFormat.Packed12: td.PackedFormat.Packed12,
      td.PackedFormat.Packed12_IDS: td.PackedFormat.Packed12_IDS,
    }

    assert self.packed_format in conversion, f'Invalid pattern: {self.packed_format}'
    decoded = td.decode12(bytes, output_dtype=torch.float32, format_type=conversion[self.packed_format])

    width, height = self.image_size
    expected_pixels = width * height
    actual_pixels = decoded.numel()

    if actual_pixels != expected_pixels:
      raise self._image_size_mismatch_error(
        f'Decoded image size mismatch: expected {expected_pixels} pixels ({width}x{height}), '
        f'got {actual_pixels} pixels.'
      )

    return decoded.view(height, width)

  @beartype
  def load_image(self, bytes: torch.Tensor) -> torch.Tensor:
    """Process a bayer image according to the pipeline settings.

    Args:
        bayer_image: Input bayer image tensor
        white_balance: White balance coefficients

    Returns:
        Processed RGB image as uint8 numpy array
    """

    bayer_image = self.load_bytes(bytes)

    return self.debayer(bayer_image)

  def debayer(self, bayer_image: torch.Tensor) -> torch.Tensor:
    assert bayer_image.ndim == 2, f'Bayer image must have 2 dimensions, got {bayer_image.shape}'

    if self.white_balance is not None:
      bayer_image = td.apply_white_balance(bayer_image, self.white_balance, self.bayer_pattern)

    if self.settings.debayer == Debayer.bilinear:
      rgb_raw = td.bilinear5x5_demosaic(bayer_image.unsqueeze(-1), self.bayer_pattern)
    elif self.settings.debayer == Debayer.rcd:
      rgb_raw = self.rcd_workspace.process(bayer_image.unsqueeze(-1))
    elif self.settings.debayer == Debayer.ppg:
      rgb_raw = self.ppg_workspace.process(bayer_image.unsqueeze(-1))
    else:
      raise AssertionError(f'Invalid debayer method: {self.settings.debayer}')

    # Postprocess
    if self.settings.postprocess:
      rgb_raw = self.postprocess_workspace.process(rgb_raw)

    return rgb_raw

  @beartype
  def process_rgb(self, rgb_raw: torch.Tensor, bounds: torch.Tensor | None = None) -> torch.Tensor:
    # Apply camera transform after debayer

    if bounds is not None:
      rgb_raw = normalize_image(rgb_raw, bounds)

    if self.settings.enable_denoise:
      rgb_raw = self.wiener_workspace.process_log_luminance(rgb_raw, self.settings.denoise)

    # Bilateral filtering
    if self.settings.enable_bilateral:
      rgb_raw = self.bil_workspace.process_rgb(rgb_raw, self.settings.bilateral)

    return rgb_raw

  @beartype
  def process(self, bytes: torch.Tensor, image_name: str) -> torch.Tensor:
    return self.process_image_set({image_name: bytes})[image_name]

  def transform(self, image: torch.Tensor, image_name: str) -> torch.Tensor:
    if isinstance(self.transforms, dict):
      return transform(image, self.transforms[image_name])

    return transform(image, self.transforms)

  @beartype
  def process_image_set(self, image_set_bytes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    image_names = list(image_set_bytes.keys())

    rgb_raw = [self.load_image(image_bytes) for image_bytes in image_set_bytes.values()]
    bounds = td.compute_image_bounds(rgb_raw, stride=8)

    self.bounds = lerp(self.bounds if self.bounds is not None else bounds, bounds, self.settings.moving_average)
    rgb_raw = [self.process_rgb(image, self.bounds) for image in rgb_raw]

    metrics = td.compute_image_metrics(rgb_raw, stride=8)
    self.metrics = lerp(self.metrics if self.metrics is not None else metrics, metrics, self.settings.moving_average)

    tonemapped = [self.tonemap(image, self.metrics) for image in rgb_raw]

    return {
      image_name: self.transform(image, image_name) for image_name, image in zip(image_names, tonemapped, strict=True)
    }

  def tonemap(self, rgb_raw: torch.Tensor, metrics: torch.Tensor | None = None) -> torch.Tensor:
    params = td.TonemapParameters(
      self.settings.tone_gamma, self.settings.tone_intensity, self.settings.light_adapt, self.settings.vibrance
    )

    if metrics is None:
      metrics = td.compute_image_metrics([rgb_raw], stride=4, min_gray=1e-4)

    # Tonemap
    match self.settings.tone_mapping:
      case ToneMapper.reinhard:
        return td.reinhard_tonemap(rgb_raw, metrics, params)
      case ToneMapper.linear:
        return td.linear_tonemap(rgb_raw, metrics, params)
      case ToneMapper.aces:
        return td.aces_tonemap(rgb_raw, params)
      case ToneMapper.adaptive_aces:
        return td.aces_tonemap(rgb_raw, params, metrics)
