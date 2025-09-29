"""Histogram display functionality for raw Bayer data analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from torch_darktable.scripts.bayer_utils import extract_bayer_channels


def get_channel_means(bayer_image, camera_settings):
  """Get mean values for RGB channels."""
  bayer_np = bayer_image.cpu().numpy()
  r_channel, g_channel, b_channel = extract_bayer_channels(bayer_np, camera_settings.bayer_pattern)

  return (float(np.mean(r_channel)), float(np.mean(g_channel)), float(np.mean(b_channel)))


def create_histograms(ax, bayer_image, camera_settings, channel_mode='all', bins=256):
  """Create histograms in a single matplotlib axis.
  
  Args:
    ax: matplotlib axis
    bayer_image: bayer image tensor
    camera_settings: camera settings
    channel_mode: 'all', 'red', 'green', or 'blue'
    bins: number of histogram bins
  """
  bayer_np = bayer_image.cpu().numpy()
  r_channel, g_channel, b_channel = extract_bayer_channels(bayer_np, camera_settings.bayer_pattern)
  
  if channel_mode == 'all':
    # Create overlaid histograms with transparency
    # Note: Green channel is normalized by 0.5 because there are 2x green pixels in Bayer pattern
    ax.hist(r_channel, bins=bins, color='red', alpha=0.6, range=(0, 1), label='Red')
    green_weights = np.full(len(g_channel), 0.5)
    ax.hist(g_channel, bins=bins, color='green', alpha=0.6, range=(0, 1), label='Green', weights=green_weights)
    ax.hist(b_channel, bins=bins, color='blue', alpha=0.6, range=(0, 1), label='Blue')
    ax.set_title('RGB Channels', color='black')
    ax.legend()
  elif channel_mode == 'red':
    ax.hist(r_channel, bins=bins, color='red', alpha=0.8, range=(0, 1))
    ax.set_title('Red Channel', color='black')
  elif channel_mode == 'green':
    # Don't normalize when showing only green channel
    ax.hist(g_channel, bins=bins, color='green', alpha=0.8, range=(0, 1))
    ax.set_title('Green Channel', color='black')
  elif channel_mode == 'blue':
    ax.hist(b_channel, bins=bins, color='blue', alpha=0.8, range=(0, 1))
    ax.set_title('Blue Channel', color='black')
  
  # Add axis labels
  ax.set_xlabel('Pixel Value', color='black')
  ylabel = 'Count (Normalized)' if channel_mode == 'all' else 'Count'
  ax.set_ylabel(ylabel, color='black')
  
  # Style the axis - white background
  ax.set_facecolor('white')
  ax.tick_params(colors='black')
  for spine in ax.spines.values():
    spine.set_color('black')
  
  # Set grid for better readability
  ax.grid(True, alpha=0.3)


def create_selective_histograms(ax, bayer_image, camera_settings, channel_states, bins=256):
  """Create histograms with selective channel display.
  
  Args:
    ax: matplotlib axis
    bayer_image: bayer image tensor  
    camera_settings: camera settings
    channel_states: dict with 'Red', 'Green', 'Blue' boolean states
    bins: number of histogram bins
  """
  bayer_np = bayer_image.cpu().numpy()
  r_channel, g_channel, b_channel = extract_bayer_channels(bayer_np, camera_settings.bayer_pattern)
  
  # Calculate saturation percentages (values >= 0.99)
  saturation_threshold = 0.99
  r_saturated = np.sum(r_channel >= saturation_threshold) / len(r_channel) * 100
  g_saturated = np.sum(g_channel >= saturation_threshold) / len(g_channel) * 100  
  b_saturated = np.sum(b_channel >= saturation_threshold) / len(b_channel) * 100
  
  # Filter out saturated pixels for histogram display (exclude 1.0 level)
  r_filtered = r_channel[r_channel < saturation_threshold]
  g_filtered = g_channel[g_channel < saturation_threshold]
  b_filtered = b_channel[b_channel < saturation_threshold]
  
  # Create histograms only for enabled channels, excluding saturated pixels
  histogram_range = (0, 0.99)  # Exclude 1.0 level
  
  if channel_states.get('Red', True):
    label = f'Red ({r_saturated:.1f}% sat)' if r_saturated > 0.1 else 'Red'
    ax.hist(r_filtered, bins=bins, color='red', alpha=0.6, range=histogram_range, label=label)
  
  if channel_states.get('Green', True):
    label = f'Green ({g_saturated:.1f}% sat)' if g_saturated > 0.1 else 'Green'
    # Divide green count by 2 since there are 2x green pixels in Bayer pattern
    green_weights = np.full(len(g_filtered), 0.5)
    ax.hist(g_filtered, bins=bins, color='green', alpha=0.6, range=histogram_range, label=label, weights=green_weights)
  
  if channel_states.get('Blue', True):
    label = f'Blue ({b_saturated:.1f}% sat)' if b_saturated > 0.1 else 'Blue'
    ax.hist(b_filtered, bins=bins, color='blue', alpha=0.6, range=histogram_range, label=label)
  
  if any(channel_states.values()):  # Only show legend if at least one channel is enabled
    ax.legend()
  ax.set_xlabel('Pixel Value (excluding saturated)')
  ax.set_ylabel('Count (Normalized)')
  ax.grid(True, alpha=0.3)


def get_channel_data(bayer_image, camera_settings):
  """Get extracted channel data for manual histogram creation."""
  bayer_np = bayer_image.cpu().numpy()
  return extract_bayer_channels(bayer_np, camera_settings.bayer_pattern)


def setup_histogram_controls_with_data(fig, histogram_ax, histogram_data, draw_callback):
  """Set up histogram channel toggle checkboxes with integrated data management.
  
  Args:
    fig: matplotlib figure
    histogram_ax: histogram axes
    histogram_data: tuple of (bayer_image, camera_settings)
    draw_callback: function to call to refresh display
    
  Returns:
    CheckButtons widget and channel states dict
  """
  # Create checkboxes for RGB channel toggles overlaid on the histogram chart
  # Position in top-right corner of the histogram display area
  checkbox_rect = [0.75, 0.85, 0.20, 0.12]  # x, y, width, height - make wider and more visible
  # Use plt.axes() for proper widget creation (not fig.add_axes)  
  ax_checkboxes = plt.axes(checkbox_rect)
  ax_checkboxes.set_xticks([])
  ax_checkboxes.set_yticks([])
  ax_checkboxes.patch.set_facecolor('white')  # Solid white background
  ax_checkboxes.patch.set_alpha(0.9)  # Almost opaque
  ax_checkboxes.set_zorder(20)  # Ensure histogram controls stay on top
  
  # Add border to make it more visible
  for spine in ax_checkboxes.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1)
  
  
  
  channel_labels = ['Red', 'Green', 'Blue'] 
  channel_states = {'Red': True, 'Green': True, 'Blue': True}
  checkbox_values = [channel_states[label] for label in channel_labels]
  
  checkboxes = CheckButtons(ax_checkboxes, channel_labels, checkbox_values)
  
  # Don't interfere with checkbox interactivity
  
  # Connect event handler with integrated data management
  def on_channel_toggle(label):
    channel_states[label] = not channel_states[label]
    bayer_image, camera_settings = histogram_data
    update_histogram_with_channels(histogram_ax, bayer_image, camera_settings, channel_states)
    draw_callback()
    
  checkboxes.on_clicked(on_channel_toggle)
  
  # CRITICAL: Force the figure to redraw and reconnect event handlers
  fig.canvas.draw_idle()
  
  return checkboxes, channel_states


def update_histogram_with_channels(ax, bayer_image, camera_settings, channel_states):
  """Update histogram display with current channel states, preserving axis limits.
  
  Args:
    ax: matplotlib axis
    bayer_image: bayer image tensor
    camera_settings: camera settings  
    channel_states: dict with 'Red', 'Green', 'Blue' boolean states
  """
  # Store current axis limits to preserve zoom/pan state
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  
  # Clear and redraw with selected channels
  ax.clear()
  
  # Create histograms with selective channels (includes green normalization)
  create_selective_histograms(ax, bayer_image, camera_settings, channel_states)
  
  # Restore axis limits
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)


