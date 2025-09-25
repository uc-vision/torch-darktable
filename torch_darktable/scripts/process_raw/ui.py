from dataclasses import replace
from pathlib import Path
from typing import get_args

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import numpy as np

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


def create_clean_axes(rect, zorder=None, visible_ticks=False, axis_off=False, for_slider=False):
  """Create axes with common cleanup options."""
  ax = plt.axes(rect)
  if not visible_ticks and not for_slider:
    ax.set_xticks([])
    ax.set_yticks([])
  if axis_off:
    ax.axis('off')
  if zorder is not None:
    ax.set_zorder(zorder)
  if for_slider:
    # Remove the crosshair cursor from slider axes
    ax.set_navigate(False)
  return ax


def create_axes_vertical(n, x=0.08, w=0.13, h=0.015, y_top=0.51, y_bottom=0.25):
  """Create vertical array of axes for sliders."""
  axes = []
  if n <= 0:
    return axes
  for i in range(n):
    t = (i / (n - 1)) if n > 1 else 0.0
    y = y_top - (y_top - y_bottom) * t
    axes.append(create_clean_axes((x, y, w, h), for_slider=True))
  return axes


def create_radio_buttons(rect, options, active_option):
  """Create radio buttons with given options and active selection."""
  ax = plt.axes(rect)
  active_index = options.index(active_option)
  return RadioButtons(ax, options, active=active_index)


class ProcessRawUI:
  """UI for the interactive raw image processing."""

  def __init__(self, image_files: list[Path], current_index: int, camera_settings,
               bayer_image, device, output_dir: Path | None = None):
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

    # JPEG quality setting
    self.jpeg_quality = 95
    self.jpeg_progressive = False

    # JPEG display state
    self.saved_jpeg_path = None
    self.show_saved_jpeg = False

    # Output directory for JPEG files
    self.output_dir = output_dir if output_dir is not None else Path('/tmp')
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Navigation settings
    self.stride = 1  # How many images to skip when navigating
    # Track modified presets - each preset can have user modifications
    # Initialize all presets with their defaults so switching always loads modified version
    self.modified_presets = {}
    for preset_name in ImagePipeline.presets:
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
    ax_prev = create_clean_axes((sidebar_x, nav_y, nav_button_w, nav_row_h))
    ax_next = create_clean_axes((sidebar_x + nav_button_w, nav_y, nav_button_w, nav_row_h))

    self.btn_prev = Button(ax_prev, '◀ Prev')
    self.btn_next = Button(ax_next, 'Next ▶')

    # Second row: stride slider
    stride_y = nav_y - nav_row_h - 0.01
    ax_stride = create_clean_axes((sidebar_x + 0.02, stride_y, sidebar_w - 0.04, nav_row_h), for_slider=True)
    self.stride_slider = Slider(ax_stride, '', 1, 50, valinit=1, valfmt='%d')  # Continuous values
    # Hide the value text to avoid layout issues
    self.stride_slider.valtext.set_visible(False)

    # Radio button groups
    available_presets = list(ImagePipeline.presets.keys())
    self.rb_presets = create_radio_buttons((sidebar_x, 0.84, sidebar_w, 0.06), available_presets, self.current_preset)

    debayer_options = get_args(Settings.__annotations__['debayer'])
    self.rb = create_radio_buttons((sidebar_x, 0.74, sidebar_w, 0.06), debayer_options, self.settings.debayer)

    tonemap_options = get_args(Settings.__annotations__['tonemap_method'])
    self.rb_tm = create_radio_buttons((sidebar_x, 0.66, sidebar_w, 0.06), tonemap_options, self.settings.tonemap_method)

    # Checkboxes (removed white_balance)
    checkbox_labels = ('postprocess', 'wiener', 'bilateral', 'laplacian')
    ax_checks = plt.axes((sidebar_x, 0.58, sidebar_w, 0.08))
    self.cb = CheckButtons(
      ax_checks,
      checkbox_labels,
      (
        self.settings.use_postprocess,
        self.settings.use_wiener,
        self.settings.use_bilateral,
        self.settings.use_laplacian,
      ),
    )
    self.checkbox_labels = checkbox_labels

    # JPEG options checkboxes (combined)
    ax_jpeg_options = plt.axes((sidebar_x, 0.135, sidebar_w, 0.04))
    self.cb_jpeg_options = CheckButtons(ax_jpeg_options, ['Show JPEG', 'Progressive'], [False, False])

    # Sliders
    (
      ax_gamma, ax_light, ax_detail, ax_bil_sigma_s, ax_bil_sigma_r,
      ax_wiener_sigma, ax_intensity, ax_vibrance, ax_lap_shadows,
      ax_lap_highlights, ax_lap_clarity
    ) = create_axes_vertical(11, x=sidebar_x + 0.06, w=sidebar_w - 0.07)

    # Tonemap group
    self.gamma = Slider(ax_gamma, 'gamma', 0.1, 3.0, valinit=self.settings.tonemap.gamma)
    self.light = Slider(ax_light, 'light_adapt', 0.0, 1.0, valinit=self.settings.tonemap.light_adapt)

    # Bilateral group
    self.detail = Slider(ax_detail, 'bil_detail', 0.0, 2.0, valinit=self.settings.bilateral_detail)
    self.bil_sigma_s = Slider(
      ax_bil_sigma_s, 'bil_sigma_s', 0.1, 20.0, valinit=self.settings.bilateral_sigma_s
    )
    self.bil_sigma_r = Slider(
      ax_bil_sigma_r, 'bil_sigma_r', 0.01, 1.0, valinit=self.settings.bilateral_sigma_r
    )

    # Wiener group
    self.wiener = Slider(ax_wiener_sigma, 'wiener_sigma', 0.001, 0.5, valinit=self.settings.wiener_sigma)

    # Color adjustment group
    self.vibrance = Slider(ax_vibrance, 'vibrance', -1.0, 1.0, valinit=self.settings.vibrance)
    self.intensity = Slider(ax_intensity, 'intensity', -1.0, 4.0, valinit=self.settings.tonemap.intensity)

    # Laplacian group
    self.lap_shadows = Slider(ax_lap_shadows, 'lap_shadows', -1.0, 3.0, valinit=self.settings.laplacian_shadows)
    self.lap_highlights = Slider(
      ax_lap_highlights, 'lap_highlights', -1.0, 3.0, valinit=self.settings.laplacian_highlights
    )
    self.lap_clarity = Slider(ax_lap_clarity, 'lap_clarity', -1.0, 1.0, valinit=self.settings.laplacian_clarity)

    # JPEG quality slider
    ax_quality = create_clean_axes((sidebar_x + 0.06, 0.12, sidebar_w - 0.07, 0.015), for_slider=True)
    self.quality_slider = Slider(ax_quality, 'jpeg_quality', 1, 100, valinit=self.jpeg_quality, valfmt='%d')

    # Filename and size info display (horizontal layout)
    ax_info = create_clean_axes((sidebar_x, 0.09, sidebar_w, 0.02), axis_off=True)
    current_path = self.image_files[self.nav_state['current_index']]
    rgb_array = self.pipeline.process(self.bayer_image)
    h, w = rgb_array.shape[:2]
    info_text = f"{current_path.name} | {w}x{h}"
    self.info_text_left = ax_info.text(
      0.02, 0.5, info_text, fontsize=8, verticalalignment='center', transform=ax_info.transAxes
    )
    self.info_text_right = ax_info.text(
      0.98, 0.5, "", fontsize=8, verticalalignment='center', horizontalalignment='right', transform=ax_info.transAxes
    )

    # Reset button at bottom of sidebar
    ax_reset = create_clean_axes((sidebar_x, 0.02, sidebar_w, 0.06))
    self.btn_reset = Button(ax_reset, 'Reset')

    # Set z-order for overlapping sliders
    ax_detail.set_zorder(10)
    ax_wiener_sigma.set_zorder(10)

    # Parameter mappings
    self.param_mappings = {
      self.gamma: nested('tonemap', 'gamma'),
      self.light: nested('tonemap', 'light_adapt'),
      self.intensity: nested('tonemap', 'intensity'),
      self.detail: field('bilateral_detail'),
      self.bil_sigma_s: field('bilateral_sigma_s'),
      self.bil_sigma_r: field('bilateral_sigma_r'),
      self.wiener: field('wiener_sigma'),
      self.vibrance: field('vibrance'),
      self.lap_shadows: field('laplacian_shadows'),
      self.lap_highlights: field('laplacian_highlights'),
      self.lap_clarity: field('laplacian_clarity'),
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
      # Update JPEG if showing it
      if self.show_saved_jpeg:
        self._update_jpeg_and_display()
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
    self.cb.set_active(3, settings_obj.use_laplacian)

  def _update_display(self, **kwargs):
    """Update the image display."""
    if kwargs:
      self.settings = replace(self.settings, **kwargs)
      # Only recreate pipeline when settings change
      self.pipeline = ImagePipeline(self.device, self.camera_settings, self.settings)
    new_img = self.pipeline.process(self.bayer_image)
    self.im.set_data(new_img)
    # Update JPEG if showing it
    if self.show_saved_jpeg:
      self._update_jpeg_and_display()
    self.fig.canvas.draw_idle()

  def _update_display_fast(self):
    """Update the image display without recreating pipeline (for navigation)."""
    if self.show_saved_jpeg and self.saved_jpeg_path and self.saved_jpeg_path.exists():
      # Show saved JPEG
      jpeg_bgr = cv2.imread(str(self.saved_jpeg_path))
      jpeg_array = cv2.cvtColor(jpeg_bgr, cv2.COLOR_BGR2RGB)
      self.im.set_data(jpeg_array)
    else:
      # Show processed image
      new_img = self.pipeline.process(self.bayer_image)
      self.im.set_data(new_img)
    # Don't call draw_idle() here - caller will handle it

  def _update_info_display(self):
    """Update the filename and size info display."""
    current_path = self.image_files[self.nav_state['current_index']]
    rgb_array = self.pipeline.process(self.bayer_image)
    h, w = rgb_array.shape[:2]
    info_text = f"{current_path.name} | {w}x{h}"
    self.info_text_left.set_text(info_text)
    self.info_text_right.set_text("")

  def _calculate_psnr(self, original, compressed):
    """Calculate PSNR between original and compressed images."""
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
      return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

  def _save_jpeg(self):
    """Save JPEG with current settings."""
    rgb_array = self.pipeline.process(self.bayer_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    # Set up JPEG encoding parameters
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)]
    if self.jpeg_progressive:
      encode_params.extend([cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    
    cv2.imwrite(str(self.saved_jpeg_path), bgr_array, encode_params)

  def _update_info_display_with_filesize(self, jpeg_filename, file_size_mb, psnr=None):
    """Update the info display with both original and JPEG filenames."""
    current_path = self.image_files[self.nav_state['current_index']]
    rgb_array = self.pipeline.process(self.bayer_image)
    h, w = rgb_array.shape[:2]
    left_text = f"{current_path.name} | {w}x{h}"
    if psnr is not None:
      right_text = f"{jpeg_filename} | {file_size_mb:.2f} MB | {psnr:.1f} dB"
    else:
      right_text = f"{jpeg_filename} | {file_size_mb:.2f} MB"
    self.info_text_left.set_text(left_text)
    self.info_text_right.set_text(right_text)
    self.fig.canvas.draw_idle()

  def _navigate_to_image(self, new_index):
    """Navigate to a specific image index."""
    if not (0 <= new_index < len(self.image_files)):
      return

    self.nav_state['current_index'] = new_index

    # Load new image but keep current settings and pipeline
    self.bayer_image = self.load_image()

    # Reset JPEG path for new image
    self.saved_jpeg_path = None

    # Update display, info, and counter, then draw once
    self._update_display_fast()
    self._update_info_display()
    
    # If showing JPEG, regenerate it for the new image
    if self.show_saved_jpeg:
      self._update_jpeg_and_display()
    
    self.fig.canvas.draw()

  def _ensure_jpeg_path(self):
    """Ensure JPEG path is set for current image."""
    if self.saved_jpeg_path is None:
      current_path = self.image_files[self.nav_state['current_index']]
      jpeg_filename = current_path.with_suffix('.jpg').name
      self.saved_jpeg_path = self.output_dir / jpeg_filename

  def _update_jpeg_and_display(self):
    """Save JPEG and update display with file info."""
    self._ensure_jpeg_path()
    
    # Get original processed image for PSNR calculation
    original_rgb = self.pipeline.process(self.bayer_image)
    
    # Save JPEG
    self._save_jpeg()
    
    # Load saved JPEG and calculate PSNR
    jpeg_bgr = cv2.imread(str(self.saved_jpeg_path))
    jpeg_rgb = cv2.cvtColor(jpeg_bgr, cv2.COLOR_BGR2RGB)
    psnr = self._calculate_psnr(original_rgb, jpeg_rgb)
    
    # Update file size info with PSNR
    file_size = self.saved_jpeg_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    self._update_info_display_with_filesize(self.saved_jpeg_path.name, file_size_mb, psnr)
    
    print(f'Saved JPEG to: {self.saved_jpeg_path} (quality: {self.jpeg_quality}, progressive: {self.jpeg_progressive}, size: {file_size_mb:.2f} MB, PSNR: {psnr:.1f} dB)')

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
      elif label == 'laplacian':
        self._update_display(use_laplacian=is_checked)
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings

    def on_rb_tm(label):
      self._update_display(tonemap_method=label)  # type: ignore[arg-type]
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings

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

    def on_quality_change(val):
      self.jpeg_quality = int(val)
      if self.show_saved_jpeg:
        self._update_jpeg_and_display()
        self._update_display_fast()
        self.fig.canvas.draw_idle()

    def on_jpeg_options_checkbox(label):
      # Update state from checkboxes
      show_jpeg, progressive = self.cb_jpeg_options.get_status()
      self.show_saved_jpeg = show_jpeg
      self.jpeg_progressive = progressive
      
      if self.show_saved_jpeg:
        self._update_jpeg_and_display()
      
      self._update_display_fast()
      self.fig.canvas.draw_idle()

    # Register all event handlers
    self.rb_presets.on_clicked(on_presets)
    self.rb.on_clicked(on_rb)
    self.rb_tm.on_clicked(on_rb_tm)
    self.stride_slider.on_changed(on_stride)
    self.quality_slider.on_changed(on_quality_change)
    self.cb.on_clicked(on_cb)
    self.cb_jpeg_options.on_clicked(on_jpeg_options_checkbox)
    self.btn_reset.on_clicked(on_reset)
    self.btn_prev.on_clicked(on_prev)
    self.btn_next.on_clicked(on_next)

  def show(self):
    """Show the UI."""
    plt.show()

  def get_current_index(self):
    """Get the current image index."""
    return self.nav_state['current_index']
