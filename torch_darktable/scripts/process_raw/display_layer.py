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
class CurrentDisplayState:
  """Current display state passed to display layer."""
  
  main_display_area: object | None  # matplotlib axes
  im: object | None  # matplotlib image object
  display_type: str | None  # 'image' or 'histogram'


@beartype
@dataclass(frozen=True)
class DisplayState:
  """Complete display state after setup."""
  
  main_display_area: object  # matplotlib axes
  im: object | None  # matplotlib image object (None for histogram mode)
  display_type: str  # 'image' or 'histogram'
  display_info: str


class DisplayLayer:
  """Display layer that handles different output modes including JPEG preview and histograms."""

  def __init__(self, base_pipeline: ImagePipeline):
    self._base_pipeline = base_pipeline
    self._user_transform = ImageTransform.none
    self._histogram_channel_mode = 'all'
    self._histogram_xlim = None
    self._histogram_ylim = None
    self._pending_histogram_data = None

  @beartype
  def setup_display(
    self,
    fig,
    bayer_image,
    camera_settings,
    mode: DisplayMode,
    current_state: CurrentDisplayState | None = None,
    user_transform=None,
    jpeg_quality: int = 95,
    jpeg_progressive: bool = False,
  ) -> DisplayState:
    """Set up complete display for the specified mode."""
    transform = user_transform if user_transform is not None else self._user_transform
    
    # Extract current state
    if current_state is None:
      current_state = CurrentDisplayState(None, None, None)
    
    main_display_area = current_state.main_display_area
    im = current_state.im
    current_display_type = current_state.display_type

    match mode:
      case DisplayMode.LEVELS:
        return self._setup_histogram_display(fig, bayer_image, camera_settings, main_display_area, current_display_type)
      case DisplayMode.JPEG:
        return self._setup_image_display(fig, main_display_area, im, current_display_type, self._process_jpeg_mode(bayer_image, transform, jpeg_quality, jpeg_progressive))
      case DisplayMode.NORMAL:
        return self._setup_image_display(fig, main_display_area, im, current_display_type, self._process_normal_mode(bayer_image, transform))


  def _process_normal_mode(self, bayer_image, transform: ImageTransform):
    """Process image for normal display mode."""
    processed = self._base_pipeline.process(bayer_image, None, transform)
    return processed, ''

  def _process_jpeg_mode(self, bayer_image, transform: ImageTransform, quality: int, progressive: bool):
    """Process image for JPEG display mode."""
    processed = self._base_pipeline.process(bayer_image, None, transform)
    jpeg_image, file_size, psnr = self._apply_jpeg_filter(processed, quality, progressive)
    file_size_mb = file_size / (1024 * 1024)
    return jpeg_image, f'{file_size_mb:.2f} MB | {psnr:.1f} dB'

  def _setup_image_display(self, fig, main_display_area, im, current_display_type, processing_result):
    """Set up image display with processed image data."""
    image, display_info = processing_result
    
    if current_display_type != 'image':
      # Setup new image display
      if main_display_area is not None:
        main_display_area.remove()
      main_display_area = fig.add_axes([0.25, 0.01, 0.74, 0.98])
      main_display_area.set_aspect('equal')
      main_display_area.axis('off')
      im = main_display_area.imshow(image, aspect='equal', interpolation='nearest')
    else:
      # Update existing image
      im.set_data(image)
      h, w = image.shape[:2]
      im.set_extent([0, w, h, 0])
    
    return DisplayState(
      main_display_area=main_display_area,
      im=im,
      display_type='image',
      display_info=display_info
    )

  def _setup_histogram_display(self, fig, bayer_image, camera_settings, main_display_area, current_display_type):
    """Set up histogram display with controls."""
    
    # Get display info
    r_mean, g_mean, b_mean = self._get_channel_means(bayer_image, camera_settings)
    display_info = f'R: μ={r_mean:.3f} | G: μ={g_mean:.3f} | B: μ={b_mean:.3f}'
    
    # Handle histogram axes setup
    if current_display_type == 'histogram' and main_display_area is not None:
      # Update existing histogram
      self._histogram_xlim = main_display_area.get_xlim()
      self._histogram_ylim = main_display_area.get_ylim()
      main_display_area.clear()
      
      create_histograms(main_display_area, bayer_image, camera_settings, self._histogram_channel_mode)
      
      if self._histogram_xlim is not None and self._histogram_ylim is not None:
        main_display_area.set_xlim(self._histogram_xlim)
        main_display_area.set_ylim(self._histogram_ylim)
      
      return DisplayState(
        main_display_area=main_display_area,
        im=None,
        display_type='histogram',
        display_info=display_info
      )
    else:
      # Create new histogram display
      if main_display_area is not None:
        main_display_area.remove()
      # Create histogram display area 
      main_display_area = fig.add_axes([0.25, 0.01, 0.74, 0.98])
      
      from .histogram_display import create_histograms
      create_histograms(main_display_area, bayer_image, camera_settings, self._histogram_channel_mode)
      
      return DisplayState(
        main_display_area=main_display_area,
        im=None,
        display_type='histogram',
        display_info=display_info
      )

  def _get_channel_means(self, bayer_image, camera_settings):
    """Get mean values for RGB channels."""
    from .histogram_display import get_channel_means
    return get_channel_means(bayer_image, camera_settings)


  @beartype
  def save_jpeg(
    self,
    bayer_image,
    current_path: Path,
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
