"""Shared UI building utilities for creating consistent layouts and widgets."""

from dataclasses import replace
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider


def create_clean_axes(rect, zorder=None, visible_ticks=False, axis_off=False, for_slider=False):
  """Create axes with clean appearance - no ticks or labels by default."""
  ax = plt.axes(rect)
  
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


def create_radio_buttons(rect, options, active_option, orientation='horizontal'):
  """Create radio buttons with given options and active selection."""
  ax = plt.axes(rect)
  ax.set_xticks([])
  ax.set_yticks([])
  
  try:
    active_index = options.index(active_option)
  except ValueError:
    active_index = 0
    
  rb = RadioButtons(ax, options, active=active_index)
  
  if orientation == 'horizontal':
    num_options = len(options)
    y_position = 0.5
    
    # Calculate positions for button + text pairs
    button_positions = []
    for i in range(num_options):
      # Each option gets 1/num_options of the width
      section_start = i / num_options
      section_width = 1 / num_options
      
      # Button at left of section, text to its right
      button_x = section_start + section_width * 0.15  # Button at 15% into section
      text_x = section_start + section_width * 0.35    # Text at 35% into section
      
      button_positions.append((button_x, y_position))
      
      # Position text alongside button
      rb.labels[i].set_position((text_x, y_position))
      rb.labels[i].set_horizontalalignment('left')
      rb.labels[i].set_verticalalignment('center')
    
    # Reposition button markers horizontally alongside text
    rb._buttons.set_offsets(button_positions)
  
  return rb


def create_checkboxes(rect, labels, values, orientation='horizontal'):
  """Create checkboxes with given labels and values, defaulting to horizontal layout."""
  if orientation == 'horizontal':
    # Create individual checkboxes in separate sub-areas of the rect
    checkboxes = []
    x, y, w, h = rect
    individual_width = w / len(labels)
    
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
      # Calculate position for this checkbox
      cb_x = x + i * individual_width
      cb_rect = (cb_x, y, individual_width, h)
      
      ax = plt.axes(cb_rect)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_zorder(10)
      
      cb = CheckButtons(ax, [label], [value])
      checkboxes.append(cb)
    
    return checkboxes
  else:
    # Vertical - use single axes with multiple labels
    ax = plt.axes(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zorder(10)
    
    return [CheckButtons(ax, labels, values)]


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
      raise ValueError(f"Not enough space for component (height={height})")
    
    rect = (self.x, self.y_current - height, self.width, height)
    self.y_current -= (height + self.gap)  # Add gap after component
    return rect

  def add_checkboxes_horizontal(self, labels, values, height=0.03):
    """Add horizontal checkboxes using automatic layout."""
    checkboxes = []
    for (label, value) in zip(labels, values, strict=True):
      # Reserve space for labels - checkboxes start further right
      checkbox_rect = self.add_component(height) 
      cb = CheckButtons(plt.axes(checkbox_rect), [label], [value])
      
      # Create axes with clean appearance
      cb.ax.set_xticks([])
      cb.ax.set_yticks([])
      checkboxes.append(cb)
    
    return checkboxes

  def add_slider_group(self, count, height_per_slider=0.015):
    """Add a group of sliders using automatic layout."""
    axes = []
    for i in range(count):
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


def field(name):
  """Helper for creating parameter field getters/setters."""
  return (lambda s: getattr(s, name), lambda s, val: replace(s, **{name: val}))


def nested(outer, inner):
  """Helper for creating nested parameter getters/setters."""
  return (
    lambda s: getattr(getattr(s, outer), inner),
    lambda s, val: replace(s, **{outer: replace(getattr(s, outer), **{inner: val})}),
  )
