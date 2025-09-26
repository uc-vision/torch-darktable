"""Histogram display functionality for raw Bayer data analysis."""

import numpy as np
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
    ax.hist(r_channel, bins=bins, color='red', alpha=0.6, range=(0, 1), label='Red')
    ax.hist(g_channel, bins=bins, color='green', alpha=0.6, range=(0, 1), label='Green')
    ax.hist(b_channel, bins=bins, color='blue', alpha=0.6, range=(0, 1), label='Blue')
    ax.set_title('RGB Channels', color='black')
    ax.legend()
  elif channel_mode == 'red':
    ax.hist(r_channel, bins=bins, color='red', alpha=0.8, range=(0, 1))
    ax.set_title('Red Channel', color='black')
  elif channel_mode == 'green':
    ax.hist(g_channel, bins=bins, color='green', alpha=0.8, range=(0, 1))
    ax.set_title('Green Channel', color='black')
  elif channel_mode == 'blue':
    ax.hist(b_channel, bins=bins, color='blue', alpha=0.8, range=(0, 1))
    ax.set_title('Blue Channel', color='black')
  
  # Add axis labels
  ax.set_xlabel('Pixel Value', color='black')
  ax.set_ylabel('Count', color='black')
  
  # Style the axis - white background
  ax.set_facecolor('white')
  ax.tick_params(colors='black')
  for spine in ax.spines.values():
    spine.set_color('black')
  
  # Set grid for better readability
  ax.grid(True, alpha=0.3)


def get_channel_data(bayer_image, camera_settings):
  """Get extracted channel data for manual histogram creation."""
  bayer_np = bayer_image.cpu().numpy()
  return extract_bayer_channels(bayer_np, camera_settings.bayer_pattern)
