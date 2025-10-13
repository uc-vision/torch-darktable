"""Image processing pipeline with configurable settings and presets."""

from beartype import beartype
import torch

import torch_darktable as td
from torch_darktable.pipeline.camera_settings import CameraSettings

from .config import Debayer, ImageProcessingSettings, ToneMapper
from .util import lerp_none, normalize_image, resize_longest_edge


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
  ):
    """Initialize the pipeline with device, camera settings and processing settings.

    Args:
        image_size: Image dimensions as (width, height)
        bayer_pattern: Bayer pattern configuration
        packed_format: Raw data packing format
        settings: Processing settings
        device: CUDA device for processing
        white_balance: White balance coefficients from camera settings (None to disable)
    """

    assert device.index is not None, f'Device not fully specified: {device}'

    self.device = device
    self.settings = settings
    self.image_size = image_size
    self.bayer_pattern = bayer_pattern
    self.packed_format = packed_format

    self.metrics: torch.Tensor | None = None
    self.bounds: torch.Tensor | None = None

    self.bil_workspace = td.create_bilateral(
      self.device, self.image_size, sigma_s=settings.bil_sigma_spatial, sigma_r=settings.bil_sigma_luminance
    )

    self.rcd_workspace = td.create_rcd(self.device, self.image_size, self.bayer_pattern)

    self.ppg_workspace = td.create_ppg(
      self.device, self.image_size, self.bayer_pattern, median_threshold=settings.ppg_median_threshold
    )

    self.postprocess_workspace = td.create_postprocess(
      self.device,
      self.image_size,
      self.bayer_pattern,
      color_smoothing_passes=settings.color_smoothing_passes,
      green_eq_local=False,
      green_eq_global=True,
      green_eq_threshold=settings.green_eq_threshold,
    )

    self.wiener_workspace = td.create_wiener(self.device, self.image_size)
    self.white_balance = torch.tensor(white_balance, device=self.device).to(torch.float32) if white_balance is not None else None

  @staticmethod
  def from_camera_settings(camera_settings: CameraSettings, device: torch.device):
    image_settings = camera_settings.preset

    return ImageProcessor(
      camera_settings.image_size,
      camera_settings.bayer_pattern,
      camera_settings.packed_format,
      image_settings,
      device=device,
      white_balance=camera_settings.white_balance,
    )

  def update_settings(self, settings: ImageProcessingSettings):
    self.settings = settings

    def changed(*attrs: str) -> bool:
      return any(getattr(self.settings, attr) != getattr(settings, attr) for attr in attrs)

    if changed('bil_sigma_spatial', 'enable_bilateral', 'bil_sigma_luminance'):
      self.bil_workspace = td.create_bilateral(
        self.device, self.image_size, sigma_s=settings.bil_sigma_spatial, sigma_r=settings.bil_sigma_luminance
      )

    self.ppg_workspace.median_threshold = settings.ppg_median_threshold
    self.postprocess_workspace.color_smoothing_passes = settings.color_smoothing_passes
    self.postprocess_workspace.green_eq_threshold = settings.green_eq_threshold

  @property
  def final_size(self):
    return resize_longest_edge(self.image_size, self.settings.resize_width)

  @beartype
  def load_bytes(self, bytes: torch.Tensor) -> torch.Tensor:
    conversion = {
      td.PackedFormat.Packed12: td.PackedFormat.Packed12,
      td.PackedFormat.Packed12_IDS: td.PackedFormat.Packed12_IDS,
    }

    assert self.packed_format in conversion, f'Invalid pattern: {self.packed_format}'
    decoded = td.decode12(bytes, output_dtype=torch.float32, format_type=conversion[self.packed_format])

    width, _ = self.image_size
    return decoded.view(-1, width)

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
      rgb_raw = td.bilateral_rgb(self.bil_workspace, rgb_raw, self.settings.bilateral)

    return rgb_raw

  @beartype
  def process(self, bytes: torch.Tensor) -> torch.Tensor:
    """Process a bayer image according to the pipeline settings.

    Args:
        bayer_image: Input bayer image tensor
        white_balance: White balance coefficients

    Returns:
        Processed RGB image as uint8 numpy array
    """

    rgb_raw = self.load_image(bytes)
    bounds = td.compute_image_bounds([rgb_raw], stride=4)

    rgb_raw = self.process_rgb(rgb_raw, bounds)
    return self.tonemap(rgb_raw)

  @beartype
  def process_image_set(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
    rgb_raw = [self.load_image(image_bytes) for image_bytes in images]
    bounds = td.compute_image_bounds(rgb_raw, stride=8)

    if self.settings.enable_temporal_average:
      self.bounds = lerp_none(self.bounds, bounds, self.settings.moving_average)
    else:
      self.bounds = bounds

    rgb_raw = [self.process_rgb(rgb_raw, self.bounds) for rgb_raw in rgb_raw]

    metrics = td.compute_image_metrics(rgb_raw, stride=8)
    if self.settings.enable_temporal_average:
      self.metrics = lerp_none(self.metrics, metrics, self.settings.moving_average)
    else:
      self.metrics = metrics

    return [self.tonemap(rgb_raw, self.metrics) for rgb_raw in rgb_raw]

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
