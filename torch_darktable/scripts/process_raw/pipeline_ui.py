"""Pipeline controller for managing settings and processing logic."""

from dataclasses import fields, replace
from pathlib import Path

from beartype import beartype
from matplotlib.widgets import Slider
import torch

import torch_darktable as td
from torch_darktable.pipeline.camera_settings import CameraSettings, load_raw_bytes_stripped
from torch_darktable.pipeline.config import Debayer, ImageProcessingSettings, ToneMapper
from torch_darktable.pipeline.image_processor import ImageProcessor
from torch_darktable.pipeline.presets import presets
from torch_darktable.pipeline.transform import ImageTransform, transform
from torch_darktable.scripts.process_raw.ui_builder import (
  UILayoutManager,
  create_checkboxes,
  create_radio_buttons,
)


class PipelineController:
  """Manages pipeline settings, processing, and image loading."""

  @beartype
  def __init__(
    self,
    camera_settings: CameraSettings,
    device: torch.device,
    image_transform: ImageTransform,
  ):
    self.camera_settings = camera_settings
    self.device = device
    self.image_transform = image_transform

    # Initialize settings with camera default preset
    self.settings = camera_settings.preset
    self.current_preset = 'default'

    # Track modified presets - each preset can have user modifications
    self.modified_presets = {'default': camera_settings.preset}
    for preset_name in presets:
      self.modified_presets[preset_name] = presets[preset_name]

    self._create_pipeline()

  def _create_pipeline(self):
    """Create the image processing pipeline."""
    self.image_processor = ImageProcessor(
      self.camera_settings.image_size,
      self.camera_settings.bayer_pattern,
      self.camera_settings.packed_format,
      self.settings,
      self.device,
      self.camera_settings.white_balance,
    )

  def _update_base_pipeline(self):
    """Update the base pipeline with current settings."""
    self.image_processor.update_settings(self.settings)


  @beartype
  def process_image(self, bayer_image: torch.Tensor) -> torch.Tensor:
    """Process a bayer image through the pipeline."""
    rgb_raw = self.image_processor.debayer(bayer_image)
    bounds = td.compute_image_bounds([rgb_raw], stride=4)

    rgb_raw = self.image_processor.process_rgb(rgb_raw, bounds)
    tonemapped = self.image_processor.tonemap(rgb_raw)
    return transform(tonemapped, self.image_transform)

  def update_display_callback(self):
    """Callback to update the main UI display when pipeline settings change."""
    # This will be set by the main UI to connect pipeline changes to display updates
    pass

  @beartype
  def load_image(self, image_path: Path) -> torch.Tensor:
    """Load an image. Return raw bayer image (unpacked and on device)"""
    bytes = load_raw_bytes_stripped(image_path, self.camera_settings, self.device)
    return self.image_processor.load_bytes(bytes)

  def switch_preset(self, preset_name: str):
    """Switch to a different preset."""
    if preset_name in self.modified_presets:
      self.current_preset = preset_name
      self.settings = self.modified_presets[preset_name]
      self._update_base_pipeline()

  def update_setting(self, setting_path: str, value):
    """Update any setting using direct field names."""
    self.settings = replace(self.settings, **{setting_path: value})
    self._update_pipeline_and_ui()

  def _update_pipeline_and_ui(self):
    """Common update logic - update pipeline, save preset, and refresh UI."""
    self._update_base_pipeline()
    self.modified_presets[self.current_preset] = self.settings
    self.update_display_callback()

  def reset_current_preset(self):
    """Reset current preset to hardcoded defaults."""
    self.settings = presets[self.current_preset]
    self.modified_presets[self.current_preset] = self.settings
    self._update_pipeline_and_ui()
    self._sync_ui_from_settings()

  def rotate_transform(self):
    """Rotate the transform to the next rotation."""
    self.image_transform = self.image_transform.next_rotation()

  def create_param_handler(self, getter, setter):
    """Create a parameter handler for sliders."""

    def handler(val):
      # Check if value actually changed to avoid unnecessary updates
      old_val = getter(self.settings)
      if old_val == val:
        return

      # Create new settings with updated value
      new_settings = setter(self.settings, val)
      self.settings = new_settings
      self._update_pipeline_and_ui()

    return handler

  def create_pipeline_ui(self, layout_manager: UILayoutManager):
    """Create all pipeline-related UI components using the layout manager."""

    # Preset radio buttons
    preset_rect = layout_manager.add_component(0.08)
    available_presets = list(presets.keys())
    self.rb_presets = create_radio_buttons(
      preset_rect, available_presets, self.current_preset, orientation='horizontal'
    )

    # Debayer method radio buttons
    debayer_rect = layout_manager.add_component(0.08)
    debayer_options = ('bilinear', 'rcd', 'ppg')
    self.rb_debayer = create_radio_buttons(
      debayer_rect, debayer_options, self.settings.debayer.name, orientation='horizontal'
    )

    # Tonemap method radio buttons
    tonemap_rect = layout_manager.add_component(0.08)
    tonemap_options = ('reinhard', 'aces', 'adaptive_aces', 'linear')
    self.rb_tonemap = create_radio_buttons(
      tonemap_rect, tonemap_options, self.settings.tone_mapping.name, orientation='horizontal'
    )

    # Checkboxes
    checkbox_labels = ('postprocess', 'denoise', 'bilateral')
    checkbox_values = (
      self.settings.postprocess,
      self.settings.enable_denoise,
      self.settings.enable_bilateral,
    )

    # Create checkboxes horizontally using consistent helper
    checkbox_rect = layout_manager.add_component(0.06)
    self.checkboxes = create_checkboxes(checkbox_rect, checkbox_labels, checkbox_values)
    self.checkbox_labels = checkbox_labels

    # Get field metadata for ranges
    field_metadata = {f.name: f.metadata for f in fields(ImageProcessingSettings)}

    # Define all potential sliders - only include those with ranges
    slider_field_definitions = [
      ('tone_gamma', lambda s: s.tone_gamma),
      ('light_adapt', lambda s: s.light_adapt),
      ('tone_intensity', lambda s: s.tone_intensity),
      ('bilateral', lambda s: s.bilateral),
      ('denoise', lambda s: s.denoise),
      ('vibrance', lambda s: s.vibrance),
    ]

    def create_setter(name):
      return lambda s, val: replace(s, **{name: val})

    slider_definitions = []
    for field_name, getter in slider_field_definitions:
      meta = field_metadata.get(field_name, {})
      # Only add sliders for fields with explicit ranges
      if 'min' in meta and 'max' in meta:
        min_val = meta['min']
        max_val = meta['max']
        setter = create_setter(field_name)
        slider_definitions.append((field_name, min_val, max_val, getter, setter))

    # Create slider axes and sliders in one loop
    slider_axes = layout_manager.add_slider_group(len(slider_definitions))
    self.param_mappings = {}

    for i, (label, min_val, max_val, getter, setter) in enumerate(slider_definitions):
      slider = Slider(slider_axes[i], label, min_val, max_val, valinit=getter(self.settings))
      self.param_mappings[slider] = (getter, setter)

    # Connect event handlers
    self._connect_pipeline_events()

  def _connect_pipeline_events(self):
    """Connect all pipeline UI event handlers."""

    # Consolidated event handlers
    def on_presets(label):
      self.switch_preset(label)
      self._sync_ui_from_settings()
      self.update_display_callback()

    def on_debayer(label):
      self.update_setting('debayer', Debayer[label])

    def on_tonemap(label):
      self.update_setting('tone_mapping', ToneMapper[label])

    def on_checkbox(label):
      idx = self.checkbox_labels.index(label)
      is_checked = self.checkboxes[idx].get_status()[0]
      # Map checkbox label to setting field
      checkbox_field_map = {'postprocess': 'postprocess', 'denoise': 'enable_denoise', 'bilateral': 'enable_bilateral'}
      self.update_setting(checkbox_field_map[label], is_checked)

    # Register handlers
    self.rb_presets.on_clicked(on_presets)
    self.rb_debayer.on_clicked(on_debayer)
    self.rb_tonemap.on_clicked(on_tonemap)

    for cb in self.checkboxes:
      cb.on_clicked(on_checkbox)

    # Register slider handlers
    for slider, (getter, setter) in self.param_mappings.items():
      slider.on_changed(self.create_param_handler(getter, setter))

  def _sync_ui_from_settings(self):
    """Update UI controls to match current settings."""
    # Update sliders
    for slider, (getter, _) in self.param_mappings.items():
      slider.set_val(getter(self.settings))

    # Update radio buttons with fallback for invalid values
    debayer_options = ('bilinear', 'rcd', 'ppg')
    try:
      debayer_index = debayer_options.index(self.settings.debayer.name)
    except (ValueError, AttributeError):
      print(f"Warning: Unknown debayer method '{self.settings.debayer}', defaulting to 'rcd'")
      debayer_index = debayer_options.index('rcd')
    self.rb_debayer.set_active(debayer_index)

    tonemap_options = ('reinhard', 'aces', 'adaptive_aces', 'linear')
    try:
      tonemap_index = tonemap_options.index(self.settings.tone_mapping.name)
    except (ValueError, AttributeError):
      print(f"Warning: Unknown tonemap method '{self.settings.tone_mapping}', defaulting to 'reinhard'")
      tonemap_index = tonemap_options.index('reinhard')
    self.rb_tonemap.set_active(tonemap_index)

    # Update checkboxes
    checkbox_values = [
      self.settings.postprocess,
      self.settings.enable_denoise,
      self.settings.enable_bilateral,
    ]
    for cb, value in zip(self.checkboxes, checkbox_values, strict=True):
      cb.set_active(0, value)
