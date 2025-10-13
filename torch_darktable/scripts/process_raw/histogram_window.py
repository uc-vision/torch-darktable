"""Histogram popup window for levels display."""

from beartype import beartype
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np

from .histogram_display import create_selective_histograms, get_channel_means


class HistogramWindow:
  """Popup window that displays histogram/levels information."""

  def __init__(self, bayer_image, camera_settings, image_processor=None):
    """Initialize histogram window."""
    self.bayer_image = bayer_image
    self.camera_settings = camera_settings
    self.image_processor = image_processor
    self.channel_states = {'Red': True, 'Green': True, 'Blue': True}

    # Create the figure and window
    self.fig = plt.figure(figsize=(8, 6), facecolor='white')
    self.fig.canvas.manager.set_window_title('Levels')  # type: ignore

    # Main histogram axes - full area
    self.hist_ax = self.fig.add_axes([0.1, 0.1, 0.85, 0.8])  # type: ignore

    # Channel controls overlaid on right side, positioned lower to avoid legend
    self.checkbox_ax = self.fig.add_axes([0.72, 0.55, 0.2, 0.15])  # type: ignore
    self.checkbox_ax.set_xticks([])
    self.checkbox_ax.set_yticks([])
    self.checkbox_ax.patch.set_facecolor('white')
    self.checkbox_ax.patch.set_alpha(0.9)

    # Add border to make it more visible
    for spine in self.checkbox_ax.spines.values():
      spine.set_color('black')
      spine.set_linewidth(1)

    # Create channel checkboxes
    self.checkboxes = CheckButtons(self.checkbox_ax, ['Red', 'Green', 'Blue'], [True, True, True])

    # Connect checkbox events
    self.checkboxes.on_clicked(self._on_channel_toggle)

    # Initial display
    self.update_display(bayer_image, camera_settings)

  @beartype
  def update_display(self, bayer_image, camera_settings, image_processor=None):
    """Update the histogram display with new image data."""
    self.bayer_image = bayer_image
    self.camera_settings = camera_settings
    if image_processor is not None:
      self.image_processor = image_processor

    # Store current axis limits to preserve zoom/pan state
    xlim = self.hist_ax.get_xlim()
    ylim = self.hist_ax.get_ylim()

    # Clear and redraw histogram
    self.hist_ax.clear()

    # Always show raw Bayer data
    create_selective_histograms(self.hist_ax, bayer_image, camera_settings, self.channel_states)

    # Add statistics info for Bayer
    r_mean, g_mean, b_mean = get_channel_means(bayer_image, camera_settings)
    info_text = f'Raw Bayer - R: μ={r_mean:.3f} | G: μ={g_mean:.3f} | B: μ={b_mean:.3f}'

    self.hist_ax.set_title(info_text)

    # Restore axis limits if they were changed from defaults
    if xlim != (0.0, 1.0) or ylim[0] != 0.0:
      self.hist_ax.set_xlim(xlim)
      self.hist_ax.set_ylim(ylim)

    # Refresh display
    self.fig.canvas.draw()  # type: ignore

  def _create_rgb_histograms(self, ax, rgb_image, bins=256):
    """Create histograms for processed RGB image data."""
    # Convert to 0-1 range for histogram
    rgb_np = rgb_image.astype(np.float32) / 255.0

    # Extract channels
    r_channel = rgb_np[:, :, 0].flatten()
    g_channel = rgb_np[:, :, 1].flatten()
    b_channel = rgb_np[:, :, 2].flatten()

    # Create histograms only for enabled channels
    if self.channel_states.get('Red', True):
      ax.hist(r_channel, bins=bins, color='red', alpha=0.6, range=(0, 1), label='Red')
    if self.channel_states.get('Green', True):
      ax.hist(g_channel, bins=bins, color='green', alpha=0.6, range=(0, 1), label='Green')
    if self.channel_states.get('Blue', True):
      ax.hist(b_channel, bins=bins, color='blue', alpha=0.6, range=(0, 1), label='Blue')

    if any(self.channel_states.values()):  # Only show legend if at least one channel is enabled
      ax.legend()
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

  def _on_channel_toggle(self, label):
    """Handle channel checkbox toggle."""
    self.channel_states[label] = not self.channel_states[label]
    self.update_display(self.bayer_image, self.camera_settings)

  def show(self):
    """Show the histogram window."""
    self.fig.show()  # type: ignore

  def close(self):
    """Close the histogram window."""
    if self.fig is not None:
      plt.close(self.fig)
      self.fig = None

  def is_open(self):
    """Check if the histogram window is open."""
    return self.fig is not None and plt.fignum_exists(self.fig.number)
