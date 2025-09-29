from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider

from torch_darktable.scripts.process_raw.display_layer import CurrentDisplayState, DisplayLayer, DisplayMode
from torch_darktable.scripts.util import load_raw_image

from .pipeline_ui import PipelineController
from .ui_builder import UILayoutManager, create_clean_axes, create_radio_buttons


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


    # Store references for navigation updates
    self.bayer_image = bayer_image
    
    # Create pipeline controller for all pipeline-related UI and logic
    self.pipeline_controller = PipelineController(
      image_files=image_files,
      current_index=current_index,
      camera_settings=camera_settings,
      device=device
    )
    
    # Create display layer using pipeline controller's base pipeline
    self.display_layer = DisplayLayer(self.pipeline_controller.base_pipeline)
    
    # Update display layer whenever pipeline changes
    def update_display_layer():
      self.display_layer._base_pipeline = self.pipeline_controller.base_pipeline
      self._update_and_draw()
      # Update histogram window if open
      if self.histogram_window is not None and self.histogram_window.is_open():
        self.histogram_window.update_display(self.bayer_image, self.camera_settings, self.pipeline_controller.base_pipeline)
      # Update JPEG preview window if open
      if self.jpeg_window is not None and self.jpeg_window.is_open():
        self.jpeg_window.base_pipeline = self.pipeline_controller.base_pipeline
        self.jpeg_window.bayer_image = self.bayer_image
        self.jpeg_window.camera_settings = self.camera_settings
        self.jpeg_window.update_display()
    self.pipeline_controller.update_display_callback = update_display_layer

    # Note: JPEG settings moved to popup window


    # Output directory for JPEG files
    self.output_dir = output_dir if output_dir is not None else Path('/tmp')
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Navigation settings
    self.stride = 1  # How many images to skip when navigating

    # UI components
    self.fig = None
    self.display_state = CurrentDisplayState(None, None, None)
    

    self._setup_ui()


  def load_image(self):
    """Load the current image and return bayer data."""
    current_path = self.image_files[self.nav_state['current_index']]
    return load_raw_image(current_path, self.camera_settings)

  def _setup_ui(self):
    """Create the matplotlib UI."""
    self.fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.01)
    
    # Display state initialized in __init__, will be set up properly in _update_and_draw

    # Set initial figure window title
    current_path = self.image_files[self.nav_state['current_index']]
    self.fig.canvas.manager.set_window_title(current_path.name)

    # Left sidebar controls
    sidebar_x = 0.02
    sidebar_w = 0.20

    # Single automatic layout manager for entire sidebar - extend bottom boundary to fit more components
    layout = UILayoutManager(sidebar_x, sidebar_w, 0.97, 0.05)
    
    # Navigation controls at top - 3 buttons in a row
    nav_rects = layout.add_button_row(3, height=0.03)
    ax_prev = create_clean_axes(nav_rects[0])
    ax_next = create_clean_axes(nav_rects[1]) 
    ax_rotate = create_clean_axes(nav_rects[2])

    self.btn_prev = Button(ax_prev, '◀ Prev')
    self.btn_next = Button(ax_next, 'Next ▶')
    self.btn_rotate = Button(ax_rotate, '↻ Rotate')

    # Stride slider
    stride_rect = layout.add_component(0.03)
    ax_stride = create_clean_axes(stride_rect, for_slider=True)
    self.stride_slider = Slider(ax_stride, 'Step', 1, 50, valinit=1, valfmt='%d')
    self.stride_slider.valtext.set_visible(False)
    
    # Create all pipeline UI components (sliders, checkboxes, radio buttons)
    self.pipeline_controller.create_pipeline_ui(layout)
    

    # Display mode controls removed - only normal mode, JPEG moved to popup

    # Add gap before bottom section
    # Add some spacing
    
    # Note: JPEG controls moved to popup window - removed from main UI
    
    # Image info display 
    info_rect = layout.add_component(0.025)
    ax_info = create_clean_axes(info_rect, axis_off=True)
    # Get dimensions from bayer image (will be updated properly in _update_and_draw)
    h, w = self.bayer_image.shape[:2]
    info_text = f'{w}x{h}'
    self.info_text_left = ax_info.text(
      0.02, 0.5, info_text, fontsize=8, verticalalignment='center', transform=ax_info.transAxes
    )
    self.info_text_right = ax_info.text(
      0.98,
      0.5,
      '',  # Will be set in _update_and_draw
      fontsize=8,
      verticalalignment='center',
      horizontalalignment='right',
      transform=ax_info.transAxes,
    )

    # Note: Histogram controls moved to popup window - removed from main UI

    # Save, reset, levels and JPEG preview buttons at bottom - 4 buttons in a row  
    button_rects = layout.add_button_row(4, height=0.06)
    ax_save = create_clean_axes(button_rects[0])
    ax_reset = create_clean_axes(button_rects[1])
    ax_levels = create_clean_axes(button_rects[2])
    ax_jpeg = create_clean_axes(button_rects[3])

    self.btn_save = Button(ax_save, 'Save JPEG')
    self.btn_reset = Button(ax_reset, 'Reset')
    self.btn_levels = Button(ax_levels, 'Show Levels')
    self.btn_jpeg = Button(ax_jpeg, 'JPEG Preview')

    # Popup window references
    self.histogram_window = None
    self.jpeg_window = None



    self._setup_event_handlers()

    # Do initial display setup
    self._update_and_draw()








  def _update_and_draw(self):
    """Update display and refresh canvas - common pattern for all callbacks."""
    self._update_display_fast()
    self.fig.canvas.draw_idle()

  def _with_update(self, callback):
    """Decorator that wraps a callback to automatically update display."""
    def wrapper(*args, **kwargs):
      callback(*args, **kwargs)
      self._update_and_draw()
    return wrapper


  def _update_display_fast(self):
    """Update the display according to the current mode."""
    current_path = self.image_files[self.nav_state['current_index']]

    # Only normal display mode
    display_mode = DisplayMode.NORMAL

    # Let display_layer handle all the logic
    # Let display layer handle complete setup
    new_display_state = self.display_layer.setup_display(
      self.fig,
      self.bayer_image,
      self.camera_settings,
      display_mode,
      current_state=self.display_state,
      jpeg_quality=95,  # Default values, not used since JPEG mode removed
      jpeg_progressive=False,
    )
    
    # Update UI state from display layer result
    self.display_state = CurrentDisplayState(
      new_display_state.main_display_area,
      new_display_state.im,
      new_display_state.display_type
    )
    

    self.fig.canvas.manager.set_window_title(current_path.name)
    self.info_text_right.set_text(new_display_state.display_info)
    self.fig.canvas.draw_idle()

  def _navigate_to_image(self, new_index):
    """Navigate to a specific image index."""
    if not (0 <= new_index < len(self.image_files)):
      return

    # Simple navigation: update index, load image, update display
    self.nav_state['current_index'] = new_index
    self.bayer_image = self.load_image()

    # Update main display (handles title, info, and image)
    self._update_and_draw()
    
    # Update histogram window if open
    if self.histogram_window is not None and self.histogram_window.is_open():
      self.histogram_window.update_display(self.bayer_image, self.camera_settings, self.pipeline_controller.base_pipeline)
    
    # Update JPEG preview window if open
    if self.jpeg_window is not None and self.jpeg_window.is_open():
      self.jpeg_window.bayer_image = self.bayer_image
      self.jpeg_window.camera_settings = self.camera_settings
      self.jpeg_window.update_display()

  def _get_jpeg_save_path(self):
    """Get the save path for current image's JPEG."""
    current_path = self.image_files[self.nav_state['current_index']]
    jpeg_filename = current_path.with_suffix('.jpg').name
    return self.output_dir / jpeg_filename

  def _setup_event_handlers(self):
    """Setup all event handlers."""
    # Parameter handlers

    # Event handlers

    def on_prev(event):
      self._navigate_to_image(self.nav_state['current_index'] - self.stride)

    def on_next(event):
      self._navigate_to_image(self.nav_state['current_index'] + self.stride)

    def on_rotate(event):
      self.display_layer.rotate_transform()
      self._update_and_draw()
      # Update JPEG preview window if open
      if self.jpeg_window is not None and self.jpeg_window.is_open():
        self.jpeg_window.update_display()

    def on_stride(val):
      self.stride = int(val)

    def on_save_jpeg(event):
      # Save JPEG to disk using display manager
      # Ensure display layer has current pipeline settings
      self.display_layer._base_pipeline = self.pipeline_controller.base_pipeline
      
      save_path = self._get_jpeg_save_path()
      current_path = self.image_files[self.nav_state['current_index']]

      size_mb = self.display_layer.save_jpeg(
        self.bayer_image, current_path, save_path, 
        user_transform=self.display_layer._user_transform,
        jpeg_quality=95, jpeg_progressive=False  # Use defaults
      )

      print(
        f'Saved JPEG to: {save_path} (quality: 95, progressive: False, size: {size_mb:.2f} MB)'
      )

    def on_reset(event):
      self.pipeline_controller.reset_current_preset()

    def on_show_levels(event):
      # Show or focus histogram popup window
      if self.histogram_window is None or not self.histogram_window.is_open():
        from .histogram_window import HistogramWindow
        self.histogram_window = HistogramWindow(self.bayer_image, self.camera_settings, self.pipeline_controller.base_pipeline)
        self.histogram_window.show()
      else:
        # Bring existing window to front and update
        self.histogram_window.fig.canvas.manager.show()
        self.histogram_window.update_display(self.bayer_image, self.camera_settings, self.pipeline_controller.base_pipeline)


    def on_show_jpeg_preview(event):
      # Show or focus JPEG preview popup window
      if self.jpeg_window is None or not self.jpeg_window.is_open():
        from .jpeg_preview_window import JpegPreviewWindow
        self.jpeg_window = JpegPreviewWindow(
          self.bayer_image, self.camera_settings, self.pipeline_controller.base_pipeline, self.display_layer
        )
        self.jpeg_window.show()
      else:
        # Bring existing window to front and update
        self.jpeg_window.fig.canvas.manager.show()
        self.jpeg_window.base_pipeline = self.pipeline_controller.base_pipeline
        self.jpeg_window.bayer_image = self.bayer_image
        self.jpeg_window.camera_settings = self.camera_settings
        self.jpeg_window.update_display()

    # Register all event handlers
    self.stride_slider.on_changed(on_stride)

    # Display mode handlers removed - only normal mode
    self.btn_save.on_clicked(on_save_jpeg)
    self.btn_reset.on_clicked(on_reset)
    self.btn_levels.on_clicked(on_show_levels)
    self.btn_jpeg.on_clicked(on_show_jpeg_preview)
    self.btn_prev.on_clicked(on_prev)
    self.btn_next.on_clicked(on_next)
    self.btn_rotate.on_clicked(on_rotate)


  def show(self):
    """Show the UI."""
    plt.show()

  def get_current_index(self):
    """Get the current image index."""
    return self.nav_state['current_index']
