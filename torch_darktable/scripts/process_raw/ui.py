from dataclasses import replace
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import numpy as np
from PIL import Image

from torch_darktable.scripts.pipeline import ImagePipeline, Settings
from torch_darktable.scripts.util import load_raw_image


def field(name):
  """Helper for creating parameter field getters/setters."""
  return (lambda s: getattr(s, name), lambda s, val: replace(s, **{name: val}))


def nested(outer, inner):
  """Helper for creating nested parameter getters/setters."""
  return (
    lambda s: getattr(getattr(s, outer), inner),
    lambda s, val: replace(s, **{outer: replace(getattr(s, outer), **{inner: val})}),
  )


class ProcessRawUI:
  """UI for the interactive raw image processing."""

  def __init__(self, image_files: list[Path], current_index: int, camera_settings,
               bayer_image, device):
    self.image_files = image_files
    self.nav_state = {'current_index': current_index}
    self.camera_settings = camera_settings
    self.device = device

    # Initialize settings with camera default preset
    self.settings = ImagePipeline.presets[camera_settings.preset]
    self.current_preset = camera_settings.preset

    # Store references for navigation updates
    self.bayer_image = bayer_image
    self.pipeline = ImagePipeline(device, camera_settings, self.settings)
    
    # Navigation settings
    self.stride = 1  # How many images to skip when navigating
    
    # Track modified presets - each preset can have user modifications
    # Initialize all presets with their defaults so switching always loads modified version
    self.modified_presets = {}
    for preset_name in ImagePipeline.presets.keys():
      self.modified_presets[preset_name] = ImagePipeline.presets[preset_name]

    # UI components
    self.fig = None
    self.ax = None
    self.im = None

    self._setup_ui()

  def load_image(self):
    """Load the current image and return bayer data."""
    current_path = self.image_files[self.nav_state['current_index']]
    return load_raw_image(current_path, self.camera_settings)

  def _setup_ui(self):
    """Create the matplotlib UI."""
    rgb_np = self.pipeline.process(self.bayer_image)

    self.fig, self.ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.01)
    self.im = self.ax.imshow(rgb_np, interpolation='nearest')
    self.ax.set_aspect('equal', adjustable='box')
    self.ax.set_axis_off()

    # Left sidebar controls
    sidebar_x = 0.02
    sidebar_w = 0.20

    # Navigation controls at top
    nav_row_h = 0.03
    nav_y = 0.96

    # Top row: prev/next buttons and counter
    nav_button_w = sidebar_w / 3
    ax_prev = plt.axes((sidebar_x, nav_y, nav_button_w, nav_row_h))
    ax_next = plt.axes((sidebar_x + nav_button_w, nav_y, nav_button_w, nav_row_h))

    self.btn_prev = Button(ax_prev, '◀ Prev')
    self.btn_next = Button(ax_next, 'Next ▶')

    # Second row: stride slider
    stride_y = nav_y - nav_row_h - 0.01
    ax_stride = plt.axes((sidebar_x + 0.02, stride_y, sidebar_w - 0.04, nav_row_h))
    self.stride_slider = Slider(ax_stride, '', 1, 50, valinit=1, valfmt='%d')  # Continuous values
    # Hide the value text to avoid layout issues
    self.stride_slider.valtext.set_visible(False)
    # Remove all ticks from stride slider axes
    ax_stride.set_xticks([])
    ax_stride.set_yticks([])

    ax_presets = plt.axes((sidebar_x, 0.84, sidebar_w, 0.06))
    available_presets = list(ImagePipeline.presets.keys())
    self.rb_presets = RadioButtons(ax_presets, available_presets,
                                   active=available_presets.index(self.current_preset))

    # Debayer method
    ax_debayer = plt.axes((sidebar_x, 0.74, sidebar_w, 0.06))
    debayer_options = get_args(Settings.__annotations__['debayer'])
    self.rb = RadioButtons(ax_debayer, debayer_options,
                           active=debayer_options.index(self.settings.debayer))

    # Tonemap method
    ax_tonemap = plt.axes((sidebar_x, 0.66, sidebar_w, 0.06))
    tonemap_options = get_args(Settings.__annotations__['tonemap_method'])
    self.rb_tm = RadioButtons(
      ax_tonemap,
      tonemap_options,
      active=tonemap_options.index(self.settings.tonemap_method),
    )

    # Checkboxes (removed white_balance)
    checkbox_labels = ('postprocess', 'wiener', 'bilateral')
    ax_checks = plt.axes((sidebar_x, 0.58, sidebar_w, 0.06))
    self.cb = CheckButtons(
      ax_checks,
      checkbox_labels,
      (self.settings.use_postprocess, self.settings.use_wiener, self.settings.use_bilateral),
    )
    self.checkbox_labels = checkbox_labels

    # Sliders
    def create_axes_vertical(
      n: int,
      x: float = sidebar_x + 0.06,
      w: float = sidebar_w - 0.07,
      h: float = 0.015,
      y_top: float = 0.51,
      y_bottom: float = 0.31,
    ):
      axes = []
      if n <= 0:
        return axes
      for i in range(n):
        t = (i / (n - 1)) if n > 1 else 0.0
        y = y_top - (y_top - y_bottom) * t
        axes.append(plt.axes((x, y, w, h)))
      return axes

    ax_gamma, ax_light, ax_detail, ax_wiener_sigma, ax_intensity, ax_vibrance = create_axes_vertical(6)

    # Tonemap group
    self.gamma = Slider(ax_gamma, 'gamma', 0.1, 3.0, valinit=self.settings.tonemap.gamma)
    self.light = Slider(ax_light, 'light_adapt', 0.0, 1.0, valinit=self.settings.tonemap.light_adapt)

    # Bilateral group
    self.detail = Slider(ax_detail, 'bil_detail', 0.0, 2.0, valinit=self.settings.bilateral_detail)

    # Wiener group
    self.wiener = Slider(ax_wiener_sigma, 'wiener_sigma', 0.001, 0.5, valinit=self.settings.wiener_sigma)

    # Color adjustment group
    self.vibrance = Slider(ax_vibrance, 'vibrance', -1.0, 1.0, valinit=self.settings.vibrance)
    self.intensity = Slider(ax_intensity, 'intensity', -1.0, 3.0, valinit=self.settings.tonemap.intensity)

    # Save and reset buttons at bottom of sidebar
    button_w = sidebar_w / 2
    ax_save = plt.axes((sidebar_x, 0.02, button_w, 0.06))
    ax_reset = plt.axes((sidebar_x + button_w, 0.02, button_w, 0.06))
    
    self.btn_save = Button(ax_save, 'Save JPEG')
    self.btn_reset = Button(ax_reset, 'Reset')
    
    # Remove ticks from button axes
    ax_save.set_xticks([])
    ax_save.set_yticks([])
    ax_reset.set_xticks([])
    ax_reset.set_yticks([])

    ax_detail.set_zorder(10)
    ax_wiener_sigma.set_zorder(10)

    # Parameter mappings
    self.param_mappings = {
      self.gamma: nested('tonemap', 'gamma'),
      self.light: nested('tonemap', 'light_adapt'),
      self.intensity: nested('tonemap', 'intensity'),
      self.detail: field('bilateral_detail'),
      self.wiener: field('wiener_sigma'),
      self.vibrance: field('vibrance'),
    }

    self._setup_event_handlers()

  def _make_param_handler(self, getter, setter):
    """Create parameter change handler."""
    def handler(val):
      self.settings = setter(self.settings, float(val))
      # Update the modified preset for current preset
      self.modified_presets[self.current_preset] = self.settings
      # Force pipeline recreation since settings changed
      self.pipeline = ImagePipeline(self.device, self.camera_settings, self.settings)
      self._update_display_fast()
      self.fig.canvas.draw_idle()
    return handler

  def _sync_ui_from_settings(self, settings_obj):
    """Update UI controls to match settings."""
    for slider, (getter, _) in self.param_mappings.items():
      slider.set_val(getter(settings_obj))

    # Update radio buttons
    debayer_options = get_args(Settings.__annotations__['debayer'])
    self.rb.set_active(debayer_options.index(settings_obj.debayer))
    tonemap_options = get_args(Settings.__annotations__['tonemap_method'])
    self.rb_tm.set_active(tonemap_options.index(settings_obj.tonemap_method))

    # Update checkboxes (no white_balance)
    self.cb.set_active(0, settings_obj.use_postprocess)
    self.cb.set_active(1, settings_obj.use_wiener)
    self.cb.set_active(2, settings_obj.use_bilateral)

  def _update_display(self, **kwargs):
    """Update the image display."""
    if kwargs:
      self.settings = replace(self.settings, **kwargs)
      # Only recreate pipeline when settings change
      self.pipeline = ImagePipeline(self.device, self.camera_settings, self.settings)
    new_img = self.pipeline.process(self.bayer_image)
    self.im.set_data(new_img)
    self.fig.canvas.draw_idle()

  def _update_display_fast(self):
    """Update the image display without recreating pipeline (for navigation)."""
    new_img = self.pipeline.process(self.bayer_image)
    self.im.set_data(new_img)
    # Don't call draw_idle() here - caller will handle it


  def _navigate_to_image(self, new_index):
    """Navigate to a specific image index."""
    if not (0 <= new_index < len(self.image_files)):
      return

    self.nav_state['current_index'] = new_index

    # Load new image but keep current settings and pipeline
    self.bayer_image = self.load_image()

    # Update display and counter, then draw once
    self._update_display_fast()
    self.fig.canvas.draw()

  def _setup_event_handlers(self):
    """Setup all event handlers."""
    # Parameter handlers
    for slider, (getter, setter) in self.param_mappings.items():
      slider.on_changed(self._make_param_handler(getter, setter))

    # Event handlers
    def on_presets(label):
      # Switch to this preset's modified version (always exists since we pre-initialize)
      self.current_preset = label
      self.settings = self.modified_presets[label]
      
      # Update pipeline and UI
      self.pipeline = ImagePipeline(self.device, self.camera_settings, self.settings)
      self._sync_ui_from_settings(self.settings)
      self._update_display_fast()
      self.fig.canvas.draw()
      print(f"Switched to {label} preset")

    def on_rb(label):
      self._update_display(debayer=label)  # type: ignore[arg-type]
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings

    def on_cb(label):
      # Get current checkbox state (after the click)
      idx = self.checkbox_labels.index(label)
      is_checked = self.cb.get_status()[idx]

      if label == 'postprocess':
        self._update_display(use_postprocess=is_checked)
      elif label == 'wiener':
        self._update_display(use_wiener=is_checked)
      elif label == 'bilateral':
        self._update_display(use_bilateral=is_checked)
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings

    def on_rb_tm(label):
      self._update_display(tonemap_method=label)  # type: ignore[arg-type]
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings

    def on_save_jpeg(event):
      # Get current processed image
      rgb_array = self.pipeline.process(self.bayer_image)

      # Convert to PIL Image and save as JPEG
      if rgb_array.dtype != np.uint8:
        rgb_array = (rgb_array * 255).astype(np.uint8)

      pil_image = Image.fromarray(rgb_array)

      # Create JPEG filename from current input path
      current_path = self.image_files[self.nav_state['current_index']]
      output_path = current_path.with_suffix('.jpg')
      pil_image.save(output_path, 'JPEG', quality=95)
      print(f'Saved JPEG to: {output_path}')

    def on_reset(event):
      # Reset current preset to hardcoded defaults
      self.settings = ImagePipeline.presets[self.current_preset]
      self.modified_presets[self.current_preset] = self.settings
      self.pipeline = ImagePipeline(self.device, self.camera_settings, self.settings)
      self._sync_ui_from_settings(self.settings)
      self._update_display_fast()
      self.fig.canvas.draw()
      print(f"Reset {self.current_preset} preset to defaults")

    def on_prev(event):
      self._navigate_to_image(self.nav_state['current_index'] - self.stride)

    def on_next(event):
      self._navigate_to_image(self.nav_state['current_index'] + self.stride)

    def on_stride(val):
      self.stride = int(val)

    # Register all event handlers
    self.rb_presets.on_clicked(on_presets)
    self.rb.on_clicked(on_rb)
    self.rb_tm.on_clicked(on_rb_tm)
    self.stride_slider.on_changed(on_stride)
    self.cb.on_clicked(on_cb)
    self.btn_save.on_clicked(on_save_jpeg)
    self.btn_reset.on_clicked(on_reset)
    self.btn_prev.on_clicked(on_prev)
    self.btn_next.on_clicked(on_next)

  def show(self):
    """Show the UI."""
    plt.show()

  def get_current_index(self):
    """Get the current image index."""
    return self.nav_state['current_index']
