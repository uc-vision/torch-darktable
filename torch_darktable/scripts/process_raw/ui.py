from dataclasses import replace
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from torch_darktable.scripts.pipeline import ImagePipeline, Settings
from torch_darktable.scripts.process_raw.display_layer import DisplayLayer, DisplayMode
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


def create_radio_buttons(rect, options, active_option, orientation='vertical'):
  """Create radio buttons with given options and active selection."""
  ax = plt.axes(rect)
  active_index = options.index(active_option)
  
  if orientation == 'horizontal':
    # For horizontal layout, we need to adjust the radio button positioning
    rb = RadioButtons(ax, options, active=active_index)
    # Adjust circle positions for horizontal layout
    num_options = len(options)
    for i, circle in enumerate(rb.circles):
      # Position circles horizontally
      x_pos = (i + 0.5) / num_options
      circle.center = (x_pos, 0.5)
    
    # Adjust label positions for horizontal layout  
    for i, label in enumerate(rb.labels):
      x_pos = (i + 0.5) / num_options
      label.set_position((x_pos, 0.2))
      label.set_horizontalalignment('center')
    
    return rb
  else:
    return RadioButtons(ax, options, active=active_index)


class ProcessRawUI:
  """UI for the interactive raw image processing."""

  def __init__(
    self,
    image_files: list[Path],
    current_index: int,
    camera_settings,
    bayer_image,
    device,
    output_dir: Path | None = None,
  ):
    self.image_files = image_files
    self.nav_state = {'current_index': current_index}
    self.camera_settings = camera_settings
    self.device = device

    # Initialize settings with camera default preset
    self.settings = ImagePipeline.presets[camera_settings.preset]
    self.current_preset = camera_settings.preset

    # Store references for navigation updates
    self.bayer_image = bayer_image
    self._create_pipeline()

    # JPEG settings
    self.jpeg_quality = 95
    self.jpeg_progressive = False

    # Histogram settings
    self.histogram_channel_mode = 'all'  # 'all', 'red', 'green', 'blue'

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
    self.main_display_area = None  # Either image axes or histogram axis
    self.current_display_type = None  # 'image' or 'histogram'
    self.im = None  # For image display

    self._setup_ui()

  def _create_pipeline(self):
    """Create the display layer with current settings."""
    base_pipeline = ImagePipeline(self.device, self.camera_settings, self.settings)
    self.display_layer = DisplayLayer(base_pipeline)

  def _update_base_pipeline(self):
    """Update the base pipeline while preserving display layer state."""
    # Replace just the base pipeline, keeping display state
    new_base_pipeline = ImagePipeline(self.device, self.camera_settings, self.settings)
    self.display_layer._base_pipeline = new_base_pipeline

  def load_image(self):
    """Load the current image and return bayer data."""
    current_path = self.image_files[self.nav_state['current_index']]
    return load_raw_image(current_path, self.camera_settings)

  def _setup_ui(self):
    """Create the matplotlib UI."""
    current_path = self.image_files[self.nav_state['current_index']]
    result = self.display_layer.process_for_display(
      self.bayer_image, self.camera_settings, DisplayMode.NORMAL
    )
    rgb_np = result.image

    self.fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.01)
    
    # Initialize with image display
    self._setup_image_display(rgb_np)

    # Set initial figure window title
    self.fig.canvas.manager.set_window_title(current_path.name)

    # Left sidebar controls
    sidebar_x = 0.02
    sidebar_w = 0.20

    # Navigation controls at top
    nav_row_h = 0.03
    nav_y = 0.96

    # Top row: prev/next buttons and rotate button
    nav_button_w = sidebar_w / 4
    ax_prev = create_clean_axes((sidebar_x, nav_y, nav_button_w, nav_row_h))
    ax_next = create_clean_axes((sidebar_x + nav_button_w, nav_y, nav_button_w, nav_row_h))
    ax_rotate = create_clean_axes((sidebar_x + 2 * nav_button_w, nav_y, nav_button_w, nav_row_h))

    self.btn_prev = Button(ax_prev, '◀ Prev')
    self.btn_next = Button(ax_next, 'Next ▶')
    self.btn_rotate = Button(ax_rotate, '↻ Rotate')

    # Second row: stride slider
    stride_y = nav_y - nav_row_h - 0.01
    ax_stride = create_clean_axes((sidebar_x + 0.02, stride_y, sidebar_w - 0.04, nav_row_h), for_slider=True)
    self.stride_slider = Slider(ax_stride, '', 1, 50, valinit=1, valfmt='%d')  # Continuous values
    # Hide the value text to avoid layout issues
    self.stride_slider.valtext.set_visible(False)

    # Radio button groups (horizontal)
    available_presets = list(ImagePipeline.presets.keys())
    self.rb_presets = create_radio_buttons((sidebar_x, 0.91, sidebar_w, 0.03), available_presets, self.current_preset, 'horizontal')

    debayer_options = get_args(Settings.__annotations__['debayer'])
    self.rb = create_radio_buttons((sidebar_x, 0.87, sidebar_w, 0.03), debayer_options, self.settings.debayer, 'horizontal')

    tonemap_options = get_args(Settings.__annotations__['tonemap_method'])
    self.rb_tm = create_radio_buttons((sidebar_x, 0.83, sidebar_w, 0.03), tonemap_options, self.settings.tonemap_method, 'horizontal')

    # Checkboxes (removed white_balance)
    checkbox_labels = ('postprocess', 'wiener', 'bilateral', 'laplacian')
    ax_checks = plt.axes((sidebar_x, 0.75, sidebar_w, 0.08))
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

    # Display mode radio buttons (horizontal)
    display_modes = ['Normal', 'JPEG', 'Levels']
    self.rb_display = create_radio_buttons((sidebar_x, 0.15, sidebar_w, 0.03), display_modes, 'Normal', 'horizontal')

    # Histogram channel selection (horizontal, only relevant in Levels mode)
    channel_modes = ['All', 'Red', 'Green', 'Blue']
    self.rb_histogram_channels = create_radio_buttons((sidebar_x, 0.11, sidebar_w, 0.03), channel_modes, 'All', 'horizontal')

    # Progressive JPEG checkbox (only relevant in JPEG mode)
    ax_progressive = plt.axes((sidebar_x, 0.07, sidebar_w, 0.02))
    self.cb_progressive = CheckButtons(ax_progressive, ['Progressive'], [False])

    # Sliders
    (
      ax_gamma,
      ax_light,
      ax_detail,
      ax_bil_sigma_s,
      ax_bil_sigma_r,
      ax_wiener_sigma,
      ax_intensity,
      ax_vibrance,
      ax_lap_shadows,
      ax_lap_highlights,
      ax_lap_clarity,
    ) = create_axes_vertical(11, x=sidebar_x + 0.06, w=sidebar_w - 0.07, y_top=0.66, y_bottom=0.18)

    # Tonemap group
    self.gamma = Slider(ax_gamma, 'gamma', 0.1, 3.0, valinit=self.settings.tonemap.gamma)
    self.light = Slider(ax_light, 'light_adapt', 0.0, 1.0, valinit=self.settings.tonemap.light_adapt)

    # Bilateral group
    self.detail = Slider(ax_detail, 'bil_detail', 0.0, 2.0, valinit=self.settings.bilateral_detail)
    self.bil_sigma_s = Slider(ax_bil_sigma_s, 'bil_sigma_s', 0.1, 20.0, valinit=self.settings.bilateral_sigma_s)
    self.bil_sigma_r = Slider(ax_bil_sigma_r, 'bil_sigma_r', 0.01, 1.0, valinit=self.settings.bilateral_sigma_r)

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

    # JPEG quality slider (only relevant in JPEG mode)
    ax_quality = create_clean_axes((sidebar_x + 0.06, 0.105, sidebar_w - 0.07, 0.015), for_slider=True)
    self.quality_slider = Slider(ax_quality, 'jpeg_quality', 1, 100, valinit=self.jpeg_quality, valfmt='%d')

    # Image info display (horizontal layout)
    ax_info = create_clean_axes((sidebar_x, 0.09, sidebar_w, 0.02), axis_off=True)
    h, w = result.image.shape[:2]
    info_text = f'{w}x{h}'
    self.info_text_left = ax_info.text(
      0.02, 0.5, info_text, fontsize=8, verticalalignment='center', transform=ax_info.transAxes
    )
    self.info_text_right = ax_info.text(
      0.98,
      0.5,
      result.display_info,
      fontsize=8,
      verticalalignment='center',
      horizontalalignment='right',
      transform=ax_info.transAxes,
    )

    # Save and reset buttons at bottom of sidebar
    button_w = sidebar_w / 2
    ax_save = create_clean_axes((sidebar_x, 0.02, button_w, 0.06))
    ax_reset = create_clean_axes((sidebar_x + button_w, 0.02, button_w, 0.06))

    self.btn_save = Button(ax_save, 'Save JPEG')
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

  def _setup_image_display(self, rgb_image):
    """Setup the main area for image display."""
    if self.main_display_area is not None:
      self.main_display_area.remove()
    
    # Create single axes for image display
    self.main_display_area = self.fig.add_axes([0.25, 0.01, 0.74, 0.98])
    self.im = self.main_display_area.imshow(rgb_image, interpolation='nearest')
    self.main_display_area.set_aspect('equal', adjustable='box')
    self.main_display_area.set_axis_off()
    self.current_display_type = 'image'

  def _setup_histogram_display(self, bayer_image, camera_settings):
    """Setup the main area for histogram display."""
    if self.main_display_area is not None:
      self.main_display_area.remove()
    self.im = None
    
    # Create single axes for overlaid histogram display
    self.main_display_area = self.fig.add_axes([0.25, 0.01, 0.74, 0.98])
    
    # Create histograms using the histogram module
    from .histogram_display import create_histograms
    create_histograms(self.main_display_area, bayer_image, camera_settings, self.histogram_channel_mode)
    
    self.current_display_type = 'histogram'

  def _get_channel_means(self, bayer_image, camera_settings):
    """Get mean values for RGB channels."""
    from .histogram_display import get_channel_means
    return get_channel_means(bayer_image, camera_settings)

  def _make_param_handler(self, getter, setter):
    """Create parameter change handler."""

    def handler(val):
      self.settings = setter(self.settings, float(val))
      # Update the modified preset for current preset
      self.modified_presets[self.current_preset] = self.settings
      # Update base pipeline while preserving display state
      self._update_base_pipeline()
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
    self.cb.set_active(3, settings_obj.use_laplacian)

  def _update_display_fast(self):
    """Update the display according to the current mode."""
    current_path = self.image_files[self.nav_state['current_index']]

    # Get display mode from radio buttons
    display_mode_text = self.rb_display.value_selected
    display_mode_map = {'Normal': DisplayMode.NORMAL, 'JPEG': DisplayMode.JPEG, 'Levels': DisplayMode.LEVELS}
    display_mode = display_mode_map.get(display_mode_text, DisplayMode.NORMAL)

    if display_mode == DisplayMode.LEVELS:
      # Switch to histogram display
      if self.current_display_type != 'histogram':
        self._setup_histogram_display(self.bayer_image, self.camera_settings)
      else:
        # Update existing histogram
        # Clear and redraw histogram
        self.main_display_area.clear()
        
        from .histogram_display import create_histograms
        create_histograms(self.main_display_area, self.bayer_image, self.camera_settings, self.histogram_channel_mode)
      
      # Update info text with channel means
      r_mean, g_mean, b_mean = self._get_channel_means(self.bayer_image, self.camera_settings)
      display_info = f'R: μ={r_mean:.3f} | G: μ={g_mean:.3f} | B: μ={b_mean:.3f}'
      
    else:
      # Process image for Normal or JPEG mode
      progressive_checked = self.cb_progressive.get_status()[0]
      result = self.display_layer.process_for_display(
        self.bayer_image,
        self.camera_settings,
        display_mode,
        jpeg_quality=self.jpeg_quality,
        jpeg_progressive=progressive_checked,
      )
      
      # Switch to image display if needed
      if self.current_display_type != 'image':
        self._setup_image_display(result.image)
      else:
        # Update existing image
        self.im.set_data(result.image)
        h, w = result.image.shape[:2]
        self.im.set_extent([0, w, h, 0])
      
      display_info = result.display_info

    self.fig.canvas.manager.set_window_title(current_path.name)
    self.info_text_right.set_text(display_info)

  def _navigate_to_image(self, new_index):
    """Navigate to a specific image index."""
    if not (0 <= new_index < len(self.image_files)):
      return

    # Simple navigation: update index, load image, update display
    self.nav_state['current_index'] = new_index
    self.bayer_image = self.load_image()

    # Update display (handles title, info, and image)
    self._update_display_fast()
    self.fig.canvas.draw()

  def _get_jpeg_save_path(self):
    """Get the save path for current image's JPEG."""
    current_path = self.image_files[self.nav_state['current_index']]
    jpeg_filename = current_path.with_suffix('.jpg').name
    return self.output_dir / jpeg_filename

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
      self._update_base_pipeline()
      self._sync_ui_from_settings(self.settings)
      self._update_display_fast()
      self.fig.canvas.draw()
      print(f'Switched to {label} preset')

    def on_rb(label):
      # Update settings first
      self.settings = replace(self.settings, debayer=label)  # type: ignore[arg-type]
      self._update_base_pipeline()
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings
      self._update_display_fast()
      self.fig.canvas.draw_idle()

    def on_cb(label):
      # Get current checkbox state (after the click)
      idx = self.checkbox_labels.index(label)
      is_checked = self.cb.get_status()[idx]

      # Update settings first
      if label == 'postprocess':
        self.settings = replace(self.settings, use_postprocess=is_checked)
      elif label == 'wiener':
        self.settings = replace(self.settings, use_wiener=is_checked)
      elif label == 'bilateral':
        self.settings = replace(self.settings, use_bilateral=is_checked)
      elif label == 'laplacian':
        self.settings = replace(self.settings, use_laplacian=is_checked)

      self._update_base_pipeline()
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings
      self._update_display_fast()
      self.fig.canvas.draw_idle()

    def on_rb_tm(label):
      # Update settings first
      self.settings = replace(self.settings, tonemap_method=label)  # type: ignore[arg-type]
      self._update_base_pipeline()
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings
      self._update_display_fast()
      self.fig.canvas.draw_idle()

    def on_reset(event):
      # Reset current preset to hardcoded defaults
      self.settings = ImagePipeline.presets[self.current_preset]
      self.modified_presets[self.current_preset] = self.settings
      self._update_base_pipeline()
      self._sync_ui_from_settings(self.settings)
      self._update_display_fast()
      self.fig.canvas.draw()
      print(f'Reset {self.current_preset} preset to defaults')

    def on_prev(event):
      self._navigate_to_image(self.nav_state['current_index'] - self.stride)

    def on_next(event):
      self._navigate_to_image(self.nav_state['current_index'] + self.stride)

    def on_rotate(event):
      self.display_layer.rotate_transform()
      self._update_display_fast()
      self.fig.canvas.draw_idle()

    def on_stride(val):
      self.stride = int(val)

    def on_quality_change(val):
      self.jpeg_quality = int(val)
      # Update display if in JPEG mode
      display_mode_text = self.rb_display.value_selected
      if display_mode_index == 1:  # JPEG mode
        self._update_display_fast()
        self.fig.canvas.draw_idle()

    def on_display_mode_change(label):
      # Update display when switching between Normal/JPEG/Levels
      self._update_display_fast()
      self.fig.canvas.draw_idle()

    def on_histogram_channel_change(label):
      # Update histogram channel mode
      channel_map = {'All': 'all', 'Red': 'red', 'Green': 'green', 'Blue': 'blue'}
      self.histogram_channel_mode = channel_map.get(label, 'all')
      
      # Update display if in levels mode
      display_mode_text = self.rb_display.value_selected
      if display_mode_text == 'Levels':
        self._update_display_fast()
        self.fig.canvas.draw_idle()

    def on_progressive_checkbox(label):
      # Update progressive setting
      progressive_checked = self.cb_progressive.get_status()[0]
      self.jpeg_progressive = progressive_checked

      # Update display if in JPEG mode
      display_mode_text = self.rb_display.value_selected
      if display_mode_index == 1:  # JPEG mode
        self._update_display_fast()
        self.fig.canvas.draw_idle()

    def on_save_jpeg(event):
      # Save JPEG to disk using display manager
      save_path = self._get_jpeg_save_path()
      current_path = self.image_files[self.nav_state['current_index']]
      progressive_checked = self.cb_progressive.get_status()[0]

      size_mb = self.display_layer.save_jpeg(
        self.bayer_image, current_path, save_path, jpeg_quality=self.jpeg_quality, jpeg_progressive=progressive_checked
      )

      print(
        f'Saved JPEG to: {save_path} (quality: {self.jpeg_quality}, progressive: {progressive_checked}, size: {size_mb:.2f} MB)'
      )

    # Slider event handlers
    def on_slider_change(val, setting_path):
      """Generic slider handler that updates nested settings."""
      if '.' in setting_path:
        # Handle nested settings like 'tonemap.gamma'
        parts = setting_path.split('.')
        if parts[0] == 'tonemap':
          tonemap = replace(self.settings.tonemap, **{parts[1]: val})
          self.settings = replace(self.settings, tonemap=tonemap)
      else:
        # Handle top-level settings
        self.settings = replace(self.settings, **{setting_path: val})

      self._update_base_pipeline()
      self.modified_presets[self.current_preset] = self.settings
      self._update_display_fast()
      self.fig.canvas.draw_idle()

    # Register all event handlers
    self.rb_presets.on_clicked(on_presets)
    self.rb.on_clicked(on_rb)
    self.rb_tm.on_clicked(on_rb_tm)
    self.stride_slider.on_changed(on_stride)
    self.quality_slider.on_changed(on_quality_change)

    # Register all parameter sliders
    self.gamma.on_changed(lambda val: on_slider_change(val, 'tonemap.gamma'))
    self.light.on_changed(lambda val: on_slider_change(val, 'tonemap.light_adapt'))
    self.intensity.on_changed(lambda val: on_slider_change(val, 'tonemap.intensity'))
    self.detail.on_changed(lambda val: on_slider_change(val, 'bilateral_detail'))
    self.bil_sigma_s.on_changed(lambda val: on_slider_change(val, 'bilateral_sigma_s'))
    self.bil_sigma_r.on_changed(lambda val: on_slider_change(val, 'bilateral_sigma_r'))
    self.wiener.on_changed(lambda val: on_slider_change(val, 'wiener_sigma'))
    self.vibrance.on_changed(lambda val: on_slider_change(val, 'vibrance'))
    self.lap_shadows.on_changed(lambda val: on_slider_change(val, 'laplacian_shadows'))
    self.lap_highlights.on_changed(lambda val: on_slider_change(val, 'laplacian_highlights'))
    self.lap_clarity.on_changed(lambda val: on_slider_change(val, 'laplacian_clarity'))

    self.cb.on_clicked(on_cb)
    self.rb_display.on_clicked(on_display_mode_change)
    self.rb_histogram_channels.on_clicked(on_histogram_channel_change)
    self.cb_progressive.on_clicked(on_progressive_checkbox)
    self.btn_save.on_clicked(on_save_jpeg)
    self.btn_reset.on_clicked(on_reset)
    self.btn_prev.on_clicked(on_prev)
    self.btn_next.on_clicked(on_next)
    self.btn_rotate.on_clicked(on_rotate)

  def show(self):
    """Show the UI."""
    plt.show()

  def get_current_index(self):
    """Get the current image index."""
    return self.nav_state['current_index']
