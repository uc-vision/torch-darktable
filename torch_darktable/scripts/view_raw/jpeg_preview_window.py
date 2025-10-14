"""JPEG preview popup window for quality/compression experimentation."""

from beartype import beartype
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np


def _apply_jpeg_filter(rgb_image, quality: int, progressive: bool):
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
  psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

  return jpeg_rgb, file_size, psnr


class JpegPreviewWindow:
  """Popup window that displays JPEG preview with quality/progressive controls."""

  def __init__(self, ui):
    """Initialize JPEG preview window."""
    self.ui = ui

    # JPEG settings
    self.jpeg_quality = 95
    self.jpeg_progressive = False

    # Create the figure and window
    self.fig = plt.figure(figsize=(10, 8), facecolor='white')
    self.fig.canvas.manager.set_window_title('JPEG Preview')  # type: ignore

    # Main image display area - leave space at bottom for controls
    self.img_ax = self.fig.add_axes([0.05, 0.25, 0.9, 0.7])  # type: ignore
    self.img_ax.set_aspect('equal')
    self.img_ax.axis('off')
    self.im = None

    # Quality slider area
    self.slider_ax = self.fig.add_axes([0.15, 0.12, 0.6, 0.04])  # type: ignore
    self.quality_slider = Slider(self.slider_ax, 'Quality', 1, 100, valinit=self.jpeg_quality, valfmt='%d')

    # Progressive checkbox area
    self.checkbox_ax = self.fig.add_axes([0.15, 0.05, 0.3, 0.05])  # type: ignore
    self.checkbox_ax.set_xticks([])
    self.checkbox_ax.set_yticks([])
    self.checkbox_ax.axis('off')

    self.progressive_checkbox = CheckButtons(self.checkbox_ax, ['Progressive'], [self.jpeg_progressive])

    # Info text area
    self.info_ax = self.fig.add_axes([0.5, 0.05, 0.4, 0.05])  # type: ignore
    self.info_ax.set_xticks([])
    self.info_ax.set_yticks([])
    self.info_ax.axis('off')
    self.info_text = self.info_ax.text(0, 0.5, '', fontsize=10, verticalalignment='center')

    # Connect events
    self.quality_slider.on_changed(self._on_quality_change)
    self.progressive_checkbox.on_clicked(self._on_progressive_toggle)

  @beartype
  def update_display(self, processed_image):
    """Update the JPEG preview with current settings and image."""
    # Apply JPEG compression
    jpeg_image, file_size, psnr = _apply_jpeg_filter(processed_image, self.jpeg_quality, self.jpeg_progressive)

    # Update image display
    if self.im is None:
      self.im = self.img_ax.imshow(jpeg_image, aspect='equal', interpolation='nearest')
    else:
      self.im.set_data(jpeg_image)
      h, w = jpeg_image.shape[:2]
      self.im.set_extent([0, w, h, 0])

    # Update info text
    file_size_mb = file_size / (1024 * 1024)
    info_text = f'{file_size_mb:.2f} MB | {psnr:.1f} dB PSNR'
    self.info_text.set_text(info_text)

    # Refresh display
    self.fig.canvas.draw()  # type: ignore

  def _on_quality_change(self, val):
    """Handle quality slider change."""
    self.jpeg_quality = int(val)
    processed = self.ui.pipeline_controller.process_image(self.ui.bayer_image)
    self.update_display(processed)

  def _on_progressive_toggle(self, label):
    """Handle progressive checkbox toggle."""
    self.jpeg_progressive = not self.jpeg_progressive
    processed = self.ui.pipeline_controller.process_image(self.ui.bayer_image)
    self.update_display(processed)

  def show(self):
    """Show the JPEG preview window."""
    self.fig.show()  # type: ignore

  def close(self):
    """Close the JPEG preview window."""
    if self.fig is not None:
      plt.close(self.fig)
      self.fig = None

  def is_open(self):
    """Check if the JPEG preview window is open."""
    return self.fig is not None and plt.fignum_exists(self.fig.number)
