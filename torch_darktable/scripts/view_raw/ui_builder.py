"""Shared UI building utilities for creating consistent layouts and widgets."""

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons


def create_clean_axes(rect, zorder=None, visible_ticks=False, axis_off=False, for_slider=False, fig=None):
  """Create axes with clean appearance - no ticks or labels by default."""
  ax = fig.add_axes(rect) if fig is not None else plt.axes(rect)

  if not visible_ticks:
    ax.set_xticks([])
    ax.set_yticks([])

  if axis_off:
    ax.axis('off')

  # Set high zorder for UI elements to ensure they stay on top
  if zorder is not None:
    ax.set_zorder(zorder)
  else:
    ax.set_zorder(10)  # Default high zorder for UI elements

  if for_slider:
    # Remove the crosshair cursor from slider axes
    ax.set_navigate(False)

  return ax


def _measure_ui_dimensions(ax, options, font_size=8):
  """Measure text widths, button size, and spacing in axes coordinates."""
  # Force axes to draw so bbox is available
  plt.gcf().canvas.draw()

  renderer = plt.gcf().canvas.get_renderer()  # type: ignore
  axes_bbox = ax.get_window_extent(renderer=renderer)

  if axes_bbox.width <= 0:
    # Fallback if axes not properly sized
    text_widths = [len(option) * 0.08 for option in options]
    button_width = 0.04
    spacing = 0.015
    return text_widths, button_width, spacing

  temp_text = ax.text(0, 0, 'test', fontsize=font_size)
  text_widths = []

  # Measure text widths
  for option in options:
    temp_text.set_text(option)
    bbox = temp_text.get_window_extent(renderer=renderer)
    width_in_axes = bbox.width / axes_bbox.width
    # Add safety margin for rendering differences
    width_in_axes *= 1.15  # 15% safety margin
    text_widths.append(width_in_axes)

  # Measure button size by creating a temporary radio button circle
  temp_text.set_text('●')  # Use bullet character as button proxy
  button_bbox = temp_text.get_window_extent(renderer=renderer)
  button_width = button_bbox.width / axes_bbox.width

  # Calculate spacing based on font size - proportional to text height
  temp_text.set_text('M')  # Use standard character for height
  text_bbox = temp_text.get_window_extent(renderer=renderer)
  text_height = text_bbox.height / axes_bbox.height
  spacing = text_height * 0.3  # Spacing proportional to text height

  temp_text.remove()
  return text_widths, button_width, spacing


def _calculate_row_width(text_widths, button_width, spacing):
  """Calculate total width needed for a row of options."""
  return sum(button_width + spacing + tw for tw in text_widths)


def _scale_font_and_widths(text_widths, total_width, target_width):
  """Scale font size and text widths to fit target width."""
  scale_factor = target_width / total_width
  font_size = max(6, int(8 * scale_factor))
  scaled_widths = [tw * scale_factor for tw in text_widths]
  return font_size, scaled_widths


def _try_two_row_layout(text_widths, button_width, spacing, target_width):
  """Check if two-row layout would work better than scaling."""
  num_options = len(text_widths)
  if num_options < 4:
    return False

  row1_count = (num_options + 1) // 2
  row1_widths = text_widths[:row1_count]
  row2_widths = text_widths[row1_count:]

  row1_width = _calculate_row_width(row1_widths, button_width, spacing)
  row2_width = _calculate_row_width(row2_widths, button_width, spacing)

  # Be more permissive for two-row layout - allow slightly wider rows
  # since two rows look better than tiny text
  two_row_threshold = min(1.0, target_width * 1.2)  # Allow 20% over target, but max 100%
  return max(row1_width, row2_width) <= two_row_threshold


def _calculate_layout_params(text_widths, button_width, spacing, target_width=0.85):
  """Calculate font size and spacing for radio button layout."""
  total_content_width = _calculate_row_width(text_widths, button_width, spacing)
  num_options = len(text_widths)

  font_size = 8

  # If content fits comfortably in single row, use it
  if total_content_width <= target_width:
    start_offset = (1.0 - total_content_width) / 2
    return font_size, text_widths, start_offset, button_width, spacing, False

  # Content too wide for single row - ALWAYS try multi-row first
  if num_options >= 4:
    two_row_works = _try_two_row_layout(text_widths, button_width, spacing, target_width)
    if two_row_works:
      return font_size, text_widths, 0, button_width, spacing, True

    # Two-row doesn't work either - force two-row anyway (better than tiny text)
    return font_size, text_widths, 0, button_width, spacing, True

  # Less than 4 options - scale font for single row as last resort
  font_size, scaled_widths = _scale_font_and_widths(text_widths, total_content_width, target_width)
  scaled_total_width = _calculate_row_width(scaled_widths, button_width, spacing)
  start_offset = (1.0 - scaled_total_width) / 2

  return font_size, scaled_widths, start_offset, button_width, spacing, False


def _position_horizontal_radio_buttons(rb, text_widths, start_offset, button_width, spacing, font_size):
  """Position radio buttons and labels horizontally in single row."""
  button_positions = []
  current_x = start_offset
  y_position = 0.5

  for i in range(len(rb.labels)):
    button_x = current_x
    text_x = current_x + button_width + spacing

    button_positions.append((button_x, y_position))

    # Position text with calculated font size
    rb.labels[i].set_position((text_x, y_position))
    rb.labels[i].set_horizontalalignment('left')
    rb.labels[i].set_verticalalignment('center')
    rb.labels[i].set_fontsize(font_size)

    # Move to next position
    current_x = text_x + text_widths[i] + spacing

  # Reposition button markers
  rb._buttons.set_offsets(button_positions)


def _position_two_row_radio_buttons(rb, text_widths, button_width, spacing, font_size, target_width=0.95):
  """Position radio buttons and labels in two rows."""
  num_options = len(rb.labels)
  row1_count = (num_options + 1) // 2  # Top row gets extra if odd number

  button_positions = []

  # Position top row
  row1_widths = text_widths[:row1_count]
  row1_total = sum(button_width + spacing + tw for tw in row1_widths)
  row1_start = (1.0 - row1_total) / 2

  current_x = row1_start
  for i in range(row1_count):
    button_x = current_x
    text_x = current_x + button_width + spacing

    button_positions.append((button_x, 0.7))  # Top row at 70%

    rb.labels[i].set_position((text_x, 0.7))
    rb.labels[i].set_horizontalalignment('left')
    rb.labels[i].set_verticalalignment('center')
    rb.labels[i].set_fontsize(font_size)

    current_x = text_x + row1_widths[i] + spacing

  # Position bottom row
  row2_widths = text_widths[row1_count:]
  row2_total = sum(button_width + spacing + tw for tw in row2_widths)
  row2_start = (1.0 - row2_total) / 2

  current_x = row2_start
  for i in range(row1_count, num_options):
    button_x = current_x
    text_x = current_x + button_width + spacing

    button_positions.append((button_x, 0.3))  # Bottom row at 30%

    rb.labels[i].set_position((text_x, 0.3))
    rb.labels[i].set_horizontalalignment('left')
    rb.labels[i].set_verticalalignment('center')
    rb.labels[i].set_fontsize(font_size)

    current_x = text_x + row2_widths[i - row1_count] + spacing

  # Reposition button markers
  rb._buttons.set_offsets(button_positions)


def create_radio_buttons(axes, options, active_option, orientation='horizontal'):
  """Create radio buttons with given options and active selection."""
  axes.set_xticks([])
  axes.set_yticks([])

  try:
    active_index = options.index(active_option)
  except ValueError:
    active_index = 0

  rb = RadioButtons(axes, options, active=active_index)

  if orientation == 'horizontal':
    text_widths, button_width, spacing = _measure_ui_dimensions(axes, options)
    result = _calculate_layout_params(text_widths, button_width, spacing)
    font_size, scaled_widths, start_offset, button_width, spacing, use_two_rows = result

    if use_two_rows:
      _position_two_row_radio_buttons(rb, scaled_widths, button_width, spacing, font_size)
    else:
      _position_horizontal_radio_buttons(rb, scaled_widths, start_offset, button_width, spacing, font_size)

  return rb


def create_checkboxes(axes, labels, values, orientation='horizontal'):
  """Create checkboxes with given labels and values, defaulting to horizontal layout."""
  if orientation == 'horizontal':
    # Create individual checkboxes in separate sub-areas of the axes
    checkboxes = []
    pos = axes.get_position()
    x, y, w, h = pos.x0, pos.y0, pos.width, pos.height
    individual_width = w / len(labels)

    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
      # Calculate position for this checkbox
      cb_x = x + i * individual_width
      cb_rect = (cb_x, y, individual_width, h)

      ax = create_clean_axes(cb_rect, zorder=10, fig=axes.figure)

      cb = CheckButtons(ax, [label], [value])
      checkboxes.append(cb)

    return checkboxes
  # Vertical - use the provided axes directly

  return [CheckButtons(axes, labels, values)]


class UILayoutManager:
  """Manages automatic vertical layout of UI components."""

  def __init__(self, x, width, y_top, y_bottom, gap=0.01):
    self.x = x
    self.width = width
    self.y_current = y_top
    self.y_bottom = y_bottom
    self.gap = gap

  def add_component(self, height):
    """Add a component and return its rect, updating current y position."""
    if self.y_current - height < self.y_bottom:
      raise ValueError(f'Not enough space for component (height={height})')

    rect = (self.x, self.y_current - height, self.width, height)
    self.y_current -= height + self.gap  # Add gap after component
    return rect

  def add_checkboxes_horizontal(self, labels, values, height=0.03):
    """Add horizontal checkboxes using automatic layout."""
    checkboxes = []
    for label, value in zip(labels, values, strict=True):
      # Reserve space for labels - checkboxes start further right
      checkbox_rect = self.add_component(height)
      cb = CheckButtons(create_clean_axes(checkbox_rect), [label], [value])
      checkboxes.append(cb)

    return checkboxes

  def add_slider_group(self, count, height_per_slider=0.015):
    """Add a group of sliders using automatic layout."""
    axes = []
    for _i in range(count):
      # Each slider gets its own automatic layout slot
      # Reserve 35% width for labels, 65% for slider
      slider_rect = self.add_component(height_per_slider)
      x, y, w, h = slider_rect

      # Create slider axes with label space reserved
      label_width = w * 0.35
      slider_x = x + label_width
      slider_width = w - label_width

      ax = create_clean_axes((slider_x, y, slider_width, h), for_slider=True)
      axes.append(ax)

    return axes

  def add_horizontal_pair(self, left_width_ratio=0.6, height=0.02):
    """Add two horizontal components and return both rects."""
    total_rect = self.add_component(height)
    x, y, w, h = total_rect
    left_rect = (x, y, w * left_width_ratio, h)
    right_rect = (x + w * left_width_ratio + 0.01, y, w * (1 - left_width_ratio) - 0.01, h)
    return left_rect, right_rect

  def add_button_row(self, count, height=0.06):
    """Add a row of buttons and return their rects."""
    total_rect = self.add_component(height)
    x, y, w, h = total_rect
    button_width = w / count

    button_rects = []
    for i in range(count):
      button_x = x + i * button_width
      button_rects.append((button_x, y, button_width, h))

    return button_rects


def create_axes_vertical(n, x=0.08, w=0.13, h=0.015, y_top=0.51, y_bottom=0.25):
  """Create vertical array of axes for sliders with proper label spacing."""
  axes = []
  if n <= 0:
    return axes

  # Reserve space for labels - sliders start further right
  label_space = w * 0.35  # 35% for labels
  slider_x = x + label_space
  slider_w = w - label_space

  for i in range(n):
    t = (i / (n - 1)) if n > 1 else 0.0
    y = y_top - (y_top - y_bottom) * t
    axes.append(create_clean_axes((slider_x, y, slider_w, h), for_slider=True))
  return axes
