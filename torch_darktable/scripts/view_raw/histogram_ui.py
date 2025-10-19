"""Histogram display component with integrated channel controls."""

from dataclasses import dataclass

from beartype import beartype
from matplotlib import pyplot as plt

from .histogram_display import create_histograms, get_channel_means
from .ui_builder import create_clean_axes, create_radio_buttons


@beartype
@dataclass(frozen=True)
class HistogramResult:
  """Result from histogram processing."""

  display_info: str
  needs_setup: bool = False


class HistogramDisplay:
  """Handles histogram display with channel switching and scale preservation."""

  def __init__(self):
    self.channel_mode = 'all'
    self._xlim = None
    self._ylim = None
    self._histogram_axes = None
    self._channel_controls_axes = None
    self._rb_histogram_channels = None

  def setup_display(self, fig, main_area_rect, bayer_image, camera_settings):
    """Setup the histogram display area with embedded channel controls."""
    # Create full histogram display area
    self._histogram_axes = fig.add_axes(main_area_rect)

    # Create histograms
    create_histograms(self._histogram_axes, bayer_image, camera_settings, self.channel_mode)

    # Create channel selection overlay in top-right corner
    self._setup_channel_controls(fig, main_area_rect)

    return HistogramResult(display_info=self._get_display_info(bayer_image, camera_settings), needs_setup=False)

  def update_display(self, bayer_image, camera_settings, channel_mode: str):
    """Update histogram with new channel mode while preserving scale."""
    if self._histogram_axes is None:
      return HistogramResult(display_info='', needs_setup=True)

    # Update channel mode
    old_channel = self.channel_mode
    self.channel_mode = channel_mode

    # Preserve scale if switching channels
    if old_channel != channel_mode and self._xlim is not None and self._ylim is not None:
      # Save current scale
      current_xlim = self._histogram_axes.get_xlim()
      current_ylim = self._histogram_axes.get_ylim()

      # Clear and redraw
      self._histogram_axes.clear()
      create_histograms(self._histogram_axes, bayer_image, camera_settings, channel_mode)

      # Restore preserved scale
      self._histogram_axes.set_xlim(current_xlim)
      self._histogram_axes.set_ylim(current_ylim)
    else:
      # First time or same channel - just redraw and save scale
      self._histogram_axes.clear()
      create_histograms(self._histogram_axes, bayer_image, camera_settings, channel_mode)
      self._xlim = self._histogram_axes.get_xlim()
      self._ylim = self._histogram_axes.get_ylim()

    return HistogramResult(display_info=self._get_display_info(bayer_image, camera_settings), needs_setup=False)

  def _setup_channel_controls(self, fig: plt.Figure, main_area_rect: tuple[float, float, float, float]):
    """Create channel selection controls overlaid on histogram."""
    # Position overlay in top-right corner
    main_left, main_bottom, main_width, main_height = main_area_rect

    # Channel controls in top-right corner
    channel_left = main_left + main_width * 0.72
    channel_bottom = main_bottom + main_height * 0.85
    channel_width = main_width * 0.26
    channel_height = main_height * 0.12

    # Create overlay axes with semi-transparent background
    self._channel_controls_axes = create_clean_axes((channel_left, channel_bottom, channel_width, channel_height))
    self._channel_controls_axes.set_navigate(False)
    self._channel_controls_axes.patch.set_facecolor('white')
    self._channel_controls_axes.patch.set_alpha(0.85)

    # Add border
    for spine in self._channel_controls_axes.spines.values():
      spine.set_color('gray')

    # Find current active channel
    channel_map = {'all': 'All', 'red': 'Red', 'green': 'Green', 'blue': 'Blue'}
    active_channel = channel_map.get(self.channel_mode, 'All')

    channel_modes = ['All', 'Red', 'Green', 'Blue']
    self._rb_histogram_channels = create_radio_buttons(
      self._channel_controls_axes, channel_modes, active_channel, 'horizontal'
    )

  def _get_display_info(self, bayer_image, camera_settings) -> str:  # noqa: PLR6301
    """Get channel statistics for display."""
    r_mean, g_mean, b_mean = get_channel_means(bayer_image, camera_settings)
    return f'R: μ={r_mean:.3f} | G: μ={g_mean:.3f} | B: μ={b_mean:.3f}'

  def get_channel_controls(self):
    """Get the channel radio button controls for event binding."""
    return self._rb_histogram_channels
