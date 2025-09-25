"""Display pipeline with JPEG preview capability."""

from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
import cv2
import numpy as np

from torch_darktable.scripts.pipeline import ImagePipeline
from torch_darktable.scripts.util import ImageTransform


@beartype
@dataclass(frozen=True)
class DisplayResult:
  """Result from display pipeline processing."""
  image: np.ndarray
  window_title: str
  display_info: str


class DisplayLayer:
  """Display layer that handles different output modes including JPEG preview."""
  
  def __init__(self, base_pipeline: ImagePipeline):
    self._base_pipeline = base_pipeline
    self._user_transform = ImageTransform.none
  
  @beartype
  def process_for_display(self, bayer_image, filepath: Path, jpeg_mode: bool = False, 
                         user_transform=None, jpeg_quality: int = 95, jpeg_progressive: bool = False) -> DisplayResult:
    """Process image for display with all parameters."""
    transform = user_transform if user_transform is not None else self._user_transform
    processed = self._base_pipeline.process(bayer_image, None, transform)
    
    if not jpeg_mode:
      return DisplayResult(
        image=processed,
        window_title=filepath.name,
        display_info=""
      )
    
    # Apply JPEG compression
    jpeg_image, file_size, psnr = self._apply_jpeg_filter(processed, jpeg_quality, jpeg_progressive)
    file_size_mb = file_size / (1024 * 1024)
    
    return DisplayResult(
      image=jpeg_image,
      window_title=filepath.with_suffix('.jpg').name,
      display_info=f"{file_size_mb:.2f} MB | {psnr:.1f} dB"
    )
  
  @beartype
  def save_jpeg(self, bayer_image, filepath: Path, save_path: Path, user_transform=None, 
                jpeg_quality: int = 95, jpeg_progressive: bool = False):
    """Save processed image as JPEG."""
    transform = user_transform if user_transform is not None else self._user_transform
    processed = self._base_pipeline.process(bayer_image, None, transform)
    
    bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if jpeg_progressive:
      encode_params.extend([cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    
    cv2.imwrite(str(save_path), bgr, encode_params)
  
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
      raise RuntimeError("JPEG encoding failed")
    
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    jpeg_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    
    # Calculate metrics
    file_size = len(encoded.tobytes())
    mse = np.mean((rgb_image.astype(np.float64) - jpeg_rgb.astype(np.float64)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    return jpeg_rgb, file_size, psnr
