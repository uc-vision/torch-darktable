"""Image processing pipeline with configurable settings and presets."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from beartype import beartype
import cv2
import numpy as np
import torch

import torch_darktable as td
from torch_darktable.local_contrast import LaplacianParams
from torch_darktable.tonemap import TonemapParameters

from .util import CameraSettings, transform, transformed_size


@beartype
@dataclass(frozen=True)
class Settings:
  """Image processing settings."""

  debayer: Literal['bilinear', 'rcd', 'ppg', 'opencv'] = 'rcd'
  tonemap_method: Literal['reinhard', 'aces', 'linear'] = 'reinhard'
  use_postprocess: bool = False
  use_bilateral: bool = False
  use_wiener: bool = False
  use_white_balance: bool = False
  use_laplacian: bool = False

  bilateral_detail: float = 0.2
  bilateral_sigma_s: float = 2.0
  bilateral_sigma_r: float = 0.2
  ppg_median_threshold: float = 0.0
  postprocess_green_eq_threshold: float = 0.04

  tonemap: TonemapParameters = field(
    default_factory=lambda: TonemapParameters(gamma=0.75, light_adapt=0.9, intensity=3.0)
  )

  wiener_sigma: float = 0.075
  vibrance: float = 0.0

  laplacian_sigma: float = 0.2
  laplacian_shadows: float = 1.0
  laplacian_highlights: float = 1.0
  laplacian_clarity: float = 0.2


@beartype
class ImagePipeline:
  """Image processing pipeline that can process images according to configurable settings."""

  presets: ClassVar[dict[str, Settings]] = {
    'reinhard': Settings(
        use_bilateral=True,
        use_postprocess=True,
        use_wiener=True),

    'aces': Settings(
      tonemap_method='aces',
      tonemap=TonemapParameters(gamma=1.5, intensity=1.0),
      use_postprocess=True,
      use_wiener=True,
      use_bilateral=True,
      vibrance=0.25,
    ),
    'uc_current': Settings(
        debayer='bilinear',
        use_wiener=False,
        use_bilateral=False),

    'pfr_current': Settings(
      debayer='opencv',
      use_wiener=False,
      use_bilateral=False,
      use_postprocess=False,
      tonemap=TonemapParameters(gamma=1.0, intensity=0.0, light_adapt=1.0)),
  }

  def __init__(self, device: torch.device, camera_settings: CameraSettings, settings: Settings):
    """Initialize the pipeline with device, camera settings and processing settings.

    Args:
        device: CUDA device for processing
        camera_settings: Camera configuration
        settings: Processing settings
    """
    self.device = device
    self.camera_settings = camera_settings
    self.settings = settings
    self.bayer_size = camera_settings.image_size

    # Calculate RGB size after camera transform
    self.rgb_size = transformed_size(self.bayer_size, camera_settings.transform)

    # Create workspaces based on enabled settings
    self.bil_workspace = None
    self.rcd_workspace = None
    self.ppg_workspace = None
    self.postprocess_workspace = None
    self.wiener_workspace = None
    self.laplacian_workspace = None

    if settings.use_bilateral:
      self.bil_workspace = td.create_bilateral(
        self.device, self.rgb_size, sigma_s=settings.bilateral_sigma_s, sigma_r=settings.bilateral_sigma_r
      )

    if settings.debayer == 'rcd':
      self.rcd_workspace = td.create_rcd(
        self.device, self.bayer_size, self.camera_settings.bayer_pattern
      )

    if settings.debayer == 'ppg':
      self.ppg_workspace = td.create_ppg(
        self.device, self.bayer_size, self.camera_settings.bayer_pattern, median_threshold=settings.ppg_median_threshold
      )

    if settings.use_postprocess:
      self.postprocess_workspace = td.create_postprocess(
        self.device,
        self.bayer_size,
        self.camera_settings.bayer_pattern,
        color_smoothing_passes=5,
        green_eq_local=True,
        green_eq_global=True,
        green_eq_threshold=settings.postprocess_green_eq_threshold,
      )

    if settings.use_wiener:
      self.wiener_workspace = td.create_wiener(self.device, self.rgb_size)

    if settings.use_laplacian:
      laplacian_params = LaplacianParams(
        sigma=settings.laplacian_sigma,
        shadows=settings.laplacian_shadows,
        highlights=settings.laplacian_highlights,
        clarity=settings.laplacian_clarity
      )
      self.laplacian_workspace = td.create_laplacian(self.device, self.rgb_size, laplacian_params)

  @beartype
  def process(self, bayer_image: torch.Tensor, white_balance: torch.Tensor | None = None) -> np.ndarray:
    """Process a bayer image according to the pipeline settings.

    Args:
        bayer_image: Input bayer image tensor
        white_balance: White balance coefficients

    Returns:
        Processed RGB image as uint8 numpy array
    """
    # Apply white balance to bayer image if enabled
    bayer_input = bayer_image
    if self.settings.use_white_balance and white_balance is not None:
      bayer_input = td.apply_white_balance(
        bayer_image, white_balance / bayer_image.max(), self.camera_settings.bayer_pattern
      )


    # Debayer
    if self.settings.debayer == 'bilinear':
      rgb_raw = td.bilinear5x5_demosaic(bayer_input.unsqueeze(-1), self.camera_settings.bayer_pattern)
    elif self.settings.debayer == 'rcd':
      assert self.rcd_workspace is not None
      rgb_raw = self.rcd_workspace.process(bayer_input.unsqueeze(-1))
    elif self.settings.debayer == 'ppg':
      assert self.ppg_workspace is not None
      rgb_raw = self.ppg_workspace.process(bayer_input.unsqueeze(-1))
    elif self.settings.debayer == 'opencv':
      np_input = (bayer_input * 65535.0).to(torch.uint16).cpu().numpy()
      rgb_raw = cv2.cvtColor(np_input, cv2.COLOR_BAYER_RGGB2RGB)
      rgb_raw = torch.from_numpy(rgb_raw).to(self.device).to(torch.float32) / 65535.0
    else:
      raise AssertionError(f'Invalid debayer method: {self.settings.debayer}')

    # Postprocess
    if self.settings.use_postprocess:
      assert self.postprocess_workspace is not None
      rgb_raw = self.postprocess_workspace.process(rgb_raw)

    bounds = td.compute_image_bounds([rgb_raw], stride=4)
    rgb_raw = (rgb_raw - bounds[0]) / (bounds[1] - bounds[0])

    # Apply camera transform after debayer
    rgb_raw = transform(rgb_raw, self.camera_settings.transform)

    if self.settings.use_wiener:
      assert self.wiener_workspace is not None
      rgb_raw = self.wiener_workspace.process_log_luminance(rgb_raw, self.settings.wiener_sigma)


    # Bilateral filtering
    if self.settings.use_bilateral:
      assert self.bil_workspace is not None
      rgb_raw = td.bilateral_rgb(self.bil_workspace, rgb_raw, self.settings.bilateral_detail)

    # Laplacian local contrast
    if self.settings.use_laplacian:
      assert self.laplacian_workspace is not None
      rgb_raw = td.local_laplacian_rgb(self.laplacian_workspace, rgb_raw)

    # Compute metrics for tonemapping
    metrics = td.compute_image_metrics([rgb_raw], stride=4, min_gray=1e-4)
    params = self.settings.tonemap

    # Tonemap
    if self.settings.tonemap_method == 'reinhard':
      rgb_tm = td.reinhard_tonemap(rgb_raw, metrics, params)
    elif self.settings.tonemap_method == 'linear':
      rgb_tm = td.linear_tonemap(rgb_raw, metrics, params).clamp(0.0, 1.0)
    elif self.settings.tonemap_method == 'aces':
      rgb_tm = td.aces_tonemap(rgb_raw, metrics, params)
    else:
      raise AssertionError(f'Invalid tonemap method: {self.settings.tonemap_method}')

    # Apply vibrance adjustment
    if self.settings.vibrance != 0.0:
      rgb_tm = td.modify_vibrance(rgb_tm, amount=self.settings.vibrance * 4)

    return (rgb_tm * 255.0).to(torch.uint8).cpu().numpy()
