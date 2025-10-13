from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import torch

from torch_darktable.scripts.util import CameraSettings

from . import jpeg_utils
from .histogram_window import HistogramWindow
from .jpeg_preview_window import JpegPreviewWindow
from .pipeline_ui import PipelineController
from .ui_builder import UILayoutManager, create_clean_axes


def create_navigation_buttons(layout):
  """Create navigation buttons (prev, next, rotate)."""
  nav_rects = layout.add_button_row(3, height=0.03)
  btn_prev = Button(create_clean_axes(nav_rects[0]), '◀ Prev')
  btn_next = Button(create_clean_axes(nav_rects[1]), 'Next ▶')
  btn_rotate = Button(create_clean_axes(nav_rects[2]), '↻ Rotate')
  return btn_prev, btn_next, btn_rotate


def create_stride_slider(layout):
  """Create stride slider for navigation."""
  stride_rect = layout.add_component(0.03)
  ax_stride = create_clean_axes(stride_rect, for_slider=True)
  stride_slider = Slider(ax_stride, 'Step', 1, 50, valinit=1, valfmt='%d')
  stride_slider.valtext.set_visible(False)
  return stride_slider


def create_info_display(layout, image_shape):
  """Create info text display showing image dimensions."""
  info_rect = layout.add_component(0.025)
  ax_info = create_clean_axes(info_rect, axis_off=True)
  h, w = image_shape[:2]
  text_left = ax_info.text(
    0.02, 0.5, f'{w}x{h}', fontsize=8, verticalalignment='center', transform=ax_info.transAxes
  )
  text_right = ax_info.text(
    0.98, 0.5, '', fontsize=8, verticalalignment='center', horizontalalignment='right', transform=ax_info.transAxes
  )
  return text_left, text_right


def create_action_buttons(layout):
  """Create bottom action buttons (save, reset, levels, jpeg)."""
  button_rects = layout.add_button_row(4, height=0.06)
  btn_save = Button(create_clean_axes(button_rects[0]), 'Save JPEG')
  btn_reset = Button(create_clean_axes(button_rects[1]), 'Reset')
  btn_levels = Button(create_clean_axes(button_rects[2]), 'Show Levels')
  btn_jpeg = Button(create_clean_axes(button_rects[3]), 'JPEG Preview')
  return btn_save, btn_reset, btn_levels, btn_jpeg


class ProcessRawUI:
  """UI for the interactive raw image processing."""

  def __init__(
    self,
    image_files: list[Path],
    current_index: int,
    pipeline_controller: PipelineController,
    output_dir: Path | None = None,
  ):
    self.image_files = image_files
    self.pipeline_controller = pipeline_controller
    self.camera_settings = pipeline_controller.camera_settings
    self.device = pipeline_controller.device
    self.current_index = current_index
    self.stride = 1

    self.output_dir = output_dir if output_dir is not None else Path('/tmp')
    self.output_dir.mkdir(parents=True, exist_ok=True)

    self.camera_name = image_files[current_index].parent.stem

    self.pipeline_controller.update_display_callback = self._on_pipeline_change

    self.bayer_image = self.pipeline_controller.load_image(image_files[current_index])
    self.processed_image: torch.Tensor | None = None

    self.histogram_window = None
    self.jpeg_window = None

    self._setup_main_display()
    self._setup_sidebar_controls()
    self._setup_event_handlers()
    self._update_and_draw()

  def _get_processed_image(self) -> torch.Tensor:
    """Get processed image, using cache if available."""
    if self.processed_image is None:
      self.processed_image = self.pipeline_controller.process_image(self.bayer_image)
    return self.processed_image

  def _on_pipeline_change(self):
    """Called when pipeline settings change."""
    self.processed_image = None
    processed = self._update_display_fast()
    self.fig.canvas.draw_idle()  # type: ignore
    if self.histogram_window is not None and self.histogram_window.is_open():
      self.histogram_window.update_display(self.bayer_image, self.camera_settings)
    if self.jpeg_window is not None and self.jpeg_window.is_open():
      self.jpeg_window.update_display(processed)

  def _setup_main_display(self):
    """Create figure and main image display area."""
    self.fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.01)

    current_path = self.image_files[self.current_index]
    self.fig.canvas.manager.set_window_title(current_path.name)  # type: ignore

    self.main_display_area = self.fig.add_axes((0.25, 0.01, 0.74, 0.98))
    self.main_display_area.set_aspect('equal')
    self.main_display_area.axis('off')
    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
    self.im = self.main_display_area.imshow(placeholder, aspect='equal', interpolation='nearest')

  def _setup_sidebar_controls(self):
    """Create all sidebar UI controls."""
    layout = UILayoutManager(0.02, 0.20, 0.97, 0.05)

    self.btn_prev, self.btn_next, self.btn_rotate = create_navigation_buttons(layout)
    self.stride_slider = create_stride_slider(layout)
    self.pipeline_controller.create_pipeline_ui(layout)
    self.info_text_left, self.info_text_right = create_info_display(layout, self.bayer_image.shape)
    self.btn_save, self.btn_reset, self.btn_levels, self.btn_jpeg = create_action_buttons(layout)

  def _update_and_draw(self):
    """Update display and refresh canvas - common pattern for all callbacks."""
    return self._update_display_fast()

  def _with_update(self, callback):
    """Decorator that wraps a callback to automatically update display."""

    def wrapper(*args, **kwargs):
      callback(*args, **kwargs)
      self._update_and_draw()

    return wrapper

  def _update_display_fast(self):
    """Update the display according to the current mode."""
    current_path = self.image_files[self.current_index]

    processed = self._get_processed_image().cpu().numpy()

    # Update matplotlib display
    self.im.set_data(processed)
    h, w = processed.shape[:2]
    self.im.set_extent((0, w, h, 0))

    self.fig.canvas.manager.set_window_title(current_path.name)  # type: ignore
    self.fig.canvas.draw_idle()  # type: ignore

    return processed

  def _navigate_to(self, new_index):
    """Navigate to a specific index with wrapping."""
    self.current_index = new_index % len(self.image_files)
    self.bayer_image = self.pipeline_controller.load_image(self.image_files[self.current_index])
    self.processed_image = None  # Invalidate cache when loading new image
    processed = self._update_and_draw()
    if self.histogram_window is not None and self.histogram_window.is_open():
      self.histogram_window.update_display(self.bayer_image, self.camera_settings)
    if self.jpeg_window is not None and self.jpeg_window.is_open():
      self.jpeg_window.update_display(processed)

  def _get_jpeg_save_path(self):
    """Get the save path for current image's JPEG."""
    current_path = self.image_files[self.current_index]
    jpeg_filename = current_path.with_suffix('.jpg').name
    return self.output_dir / jpeg_filename

  def _setup_event_handlers(self):
    """Setup all event handlers."""
    # Parameter handlers

    # Event handlers

    def on_prev(event):
      self._navigate_to(self.current_index - self.stride)

    def on_next(event):
      self._navigate_to(self.current_index + self.stride)

    def on_rotate(event):
      self.pipeline_controller.rotate_transform()
      self.processed_image = None
      processed = self._update_and_draw()
      if self.jpeg_window is not None and self.jpeg_window.is_open():
        self.jpeg_window.update_display(processed)

    def on_stride(val):
      self.stride = int(val)

    def on_save_jpeg(event):
      save_path = self._get_jpeg_save_path()
      processed = self._get_processed_image()

      # Save using jpeg_utils
      jpeg_utils.save_jpeg_to_disk(processed, save_path, quality=95, progressive=False)

      size_mb = save_path.stat().st_size / (1024 * 1024)
      print(f'Saved JPEG to: {save_path} (quality: 95, progressive: False, size: {size_mb:.2f} MB)')

    def on_reset(event):
      self.pipeline_controller.reset_current_preset()

    def on_show_levels(event):
      if self.histogram_window is None or not self.histogram_window.is_open():
        self.histogram_window = HistogramWindow(self.bayer_image, self.camera_settings)
        self.histogram_window.show()
      else:
        self.histogram_window.fig.canvas.manager.show()
        self.histogram_window.update_display(self.bayer_image, self.camera_settings)

    def on_show_jpeg_preview(event):
      processed = self._get_processed_image()

      if self.jpeg_window is None or not self.jpeg_window.is_open():
        self.jpeg_window = JpegPreviewWindow(self)
        self.jpeg_window.update_display(processed)
        self.jpeg_window.show()
      else:
        self.jpeg_window.fig.canvas.manager.show()
        self.jpeg_window.update_display(processed)

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

  def show(self):  # noqa: PLR6301
    """Show the UI."""
    plt.show()

  def get_current_index(self):
    """Get the current image index."""
    return self.current_index
