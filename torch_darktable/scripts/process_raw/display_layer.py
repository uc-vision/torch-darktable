"""Display layer with support for multiple display modes."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from beartype import beartype
import cv2
import numpy as np

from torch_darktable.scripts.pipeline import ImagePipeline
from torch_darktable.scripts.util import ImageTransform


class DisplayMode(Enum):
  """Available display modes."""

  NORMAL = 'Normal'
  JPEG = 'JPEG'
  LEVELS = 'Levels'


@beartype
@dataclass(frozen=True)
class DisplayResult:
  """Result from display layer processing."""

  image: np.ndarray
  display_info: str


class DisplayLayer:
  """Display layer that handles different output modes including JPEG preview and histograms."""

  def __init__(self, base_pipeline: ImagePipeline):
    self._base_pipeline = base_pipeline
    self._user_transform = ImageTransform.none

  @beartype
  def process_for_display(
    self,
    bayer_image,
    camera_settings,
    mode: DisplayMode,
    user_transform=None,
    jpeg_quality: int = 95,
    jpeg_progressive: bool = False,
  ) -> DisplayResult:
    """Process image for display according to the specified mode."""
    transform = user_transform if user_transform is not None else self._user_transform

    match mode:
      case DisplayMode.LEVELS:
        # Levels mode is now handled directly in the UI
        raise ValueError("Levels mode should be handled directly in the UI, not in display_layer")
      case DisplayMode.JPEG:
        return self._process_jpeg_mode(bayer_image, transform, jpeg_quality, jpeg_progressive)
      case DisplayMode.NORMAL:
        return self._process_normal_mode(bayer_image, transform)


  def _process_normal_mode(self, bayer_image, transform: ImageTransform) -> DisplayResult:
    """Process image for normal display mode."""
    processed = self._base_pipeline.process(bayer_image, None, transform)
    return DisplayResult(image=processed, display_info='')

  def _process_jpeg_mode(
    self, bayer_image, transform: ImageTransform, quality: int, progressive: bool
  ) -> DisplayResult:
    """Process image for JPEG display mode."""
    processed = self._base_pipeline.process(bayer_image, None, transform)
    jpeg_image, file_size, psnr = self._apply_jpeg_filter(processed, quality, progressive)
    file_size_mb = file_size / (1024 * 1024)

    return DisplayResult(
      image=jpeg_image,
      display_info=f'{file_size_mb:.2f} MB | {psnr:.1f} dB',
    )


  @beartype
  def save_jpeg(
    self,
    bayer_image,
    save_path: Path,
    user_transform=None,
    jpeg_quality: int = 95,
    jpeg_progressive: bool = False,
  ):
    """Save processed image as JPEG."""
    transform = user_transform if user_transform is not None else self._user_transform
    processed = self._base_pipeline.process(bayer_image, None, transform)

    bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if jpeg_progressive:
      encode_params.extend([cv2.IMWRITE_JPEG_PROGRESSIVE, 1])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), bgr, encode_params)

    # Return file size in MB
    file_size = save_path.stat().st_size
    return file_size / (1024 * 1024)

  def rotate_transform(self) -> ImageTransform:
    """Rotate the current transform and return the new one."""
    self._user_transform = self._user_transform.next_rotation()
    return self._user_transform

  def _apply_jpeg_filter(self, rgb_image, quality: int, progressive: bool) -> tuple[np.ndarray, int, float]:
    """Apply JPEG compression and return (image, file_size, psnr)."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if progressive:
      encode_params.extend([cv2.IMWRITE_JPEG_PROGRESSIVE, 1])

    success, encoded = cv2.imencode('.jpg', bgr, encode_params)
    if not success:
      raise RuntimeError('JPEG encoding failed')

    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    jpeg_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    # Calculate metrics
    file_size = len(encoded.tobytes())
    mse = np.mean((rgb_image.astype(np.float64) - jpeg_rgb.astype(np.float64)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    return jpeg_rgb, file_size, psnr
