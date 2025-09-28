"""Pipeline controller for managing settings and processing logic."""

from dataclasses import replace
from pathlib import Path
from beartype import beartype
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider

from torch_darktable.scripts.pipeline import ImagePipeline
from torch_darktable.scripts.util import load_raw_image
from torch_darktable.scripts.process_raw.ui_builder import UILayoutManager, create_radio_buttons, create_checkboxes, field, nested


class PipelineController:
  """Manages pipeline settings, processing, and image loading."""

  def __init__(
    self,
    image_files: list[Path],
    current_index: int,
    camera_settings,
    device,
  ):
    self.image_files = image_files
    self.nav_state = {'current_index': current_index}
    self.camera_settings = camera_settings
    self.device = device

    # Initialize settings with camera default preset
    self.settings = ImagePipeline.presets[camera_settings.preset]
    self.current_preset = camera_settings.preset

    # Track modified presets - each preset can have user modifications
    self.modified_presets = {}
    for preset_name in ImagePipeline.presets:
      self.modified_presets[preset_name] = ImagePipeline.presets[preset_name]

    # Navigation settings
    self.stride = 1  # How many images to skip when navigating

    self._create_pipeline()

  def _create_pipeline(self):
    """Create the image processing pipeline."""
    self.base_pipeline = ImagePipeline(self.device, self.camera_settings, self.settings)

  def _update_base_pipeline(self):
    """Update the base pipeline with current settings."""
    self.base_pipeline.settings = self.settings
    
  def update_display_callback(self):
    """Callback to update the main UI display when pipeline settings change."""
    # This will be set by the main UI to connect pipeline changes to display updates
    pass

  @beartype
  def load_image(self) -> torch.Tensor:
    """Load the current image."""
    image_path = self.image_files[self.nav_state['current_index']]
    # Use the existing utility function
    return load_raw_image(image_path, self.camera_settings)

  def navigate_to_image(self, new_index: int) -> torch.Tensor | None:
    """Navigate to a specific image index and return the loaded image."""
    if not (0 <= new_index < len(self.image_files)):
      return None

    self.nav_state['current_index'] = new_index
    return self.load_image()

  def navigate_prev(self) -> torch.Tensor | None:
    """Navigate to previous image."""
    new_index = max(0, self.nav_state['current_index'] - self.stride)
    return self.navigate_to_image(new_index)

  def navigate_next(self) -> torch.Tensor | None:
    """Navigate to next image."""
    new_index = min(len(self.image_files) - 1, self.nav_state['current_index'] + self.stride)
    return self.navigate_to_image(new_index)

  def switch_preset(self, preset_name: str):
    """Switch to a different preset."""
    if preset_name in self.modified_presets:
      self.current_preset = preset_name
      self.settings = self.modified_presets[preset_name]
      self._update_base_pipeline()

  def update_setting(self, setting_path: str, value):
    """Update a specific setting using dot notation."""
    if '.' in setting_path:
      # Handle nested settings like 'tonemap.gamma'
      parts = setting_path.split('.')
      if parts[0] == 'tonemap':
        tonemap = replace(self.settings.tonemap, **{parts[1]: value})
        self.settings = replace(self.settings, tonemap=tonemap)
    else:
      # Handle top-level settings
      self.settings = replace(self.settings, **{setting_path: value})

    self._update_base_pipeline()
    self.modified_presets[self.current_preset] = self.settings

  def update_debayer_method(self, method: str):
    """Update debayer method."""
    self.settings = replace(self.settings, debayer=method)  # type: ignore[arg-type]
    self._update_base_pipeline()
    self.modified_presets[self.current_preset] = self.settings

  def update_tonemap_method(self, method: str):
    """Update tonemap method."""
    self.settings = replace(self.settings, tonemap_method=method)  # type: ignore[arg-type]
    self._update_base_pipeline()
    self.modified_presets[self.current_preset] = self.settings

  def update_checkbox_setting(self, setting_name: str, value: bool):
    """Update a checkbox setting."""
    if setting_name == 'postprocess':
      self.settings = replace(self.settings, use_postprocess=value)
    elif setting_name == 'wiener':
      self.settings = replace(self.settings, use_wiener=value)
    elif setting_name == 'bilateral':
      self.settings = replace(self.settings, use_bilateral=value)
    elif setting_name == 'laplacian':
      self.settings = replace(self.settings, use_laplacian=value)

    self._update_base_pipeline()
    self.modified_presets[self.current_preset] = self.settings

  def reset_current_preset(self):
    """Reset current preset to hardcoded defaults."""
    self.settings = ImagePipeline.presets[self.current_preset]
    self.modified_presets[self.current_preset] = self.settings
    self._update_base_pipeline()

  def get_current_image_path(self) -> Path:
    """Get the current image path."""
    return self.image_files[self.nav_state['current_index']]

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
      
      self._update_base_pipeline()
      # Update modified preset
      self.modified_presets[self.current_preset] = self.settings
      # Update the main UI display
      self.update_display_callback()
    
    return handler

  def create_pipeline_ui(self, layout_manager: UILayoutManager):
    """Create all pipeline-related UI components using the layout manager."""
    
    # Preset radio buttons
    preset_rect = layout_manager.add_component(0.06)
    available_presets = list(ImagePipeline.presets.keys())
    self.rb_presets = create_radio_buttons(preset_rect, available_presets, self.current_preset, orientation='horizontal')
    
    # Debayer method radio buttons  
    debayer_rect = layout_manager.add_component(0.06)
    self.rb_debayer = create_radio_buttons(debayer_rect, ('bilinear', 'rcd', 'ppg', 'opencv'), self.settings.debayer, orientation='horizontal')
    
    # Tonemap method radio buttons
    tonemap_rect = layout_manager.add_component(0.06) 
    self.rb_tonemap = create_radio_buttons(tonemap_rect, ('reinhard', 'aces', 'adaptive_aces', 'linear'), self.settings.tonemap_method, orientation='horizontal')
    
    # Checkboxes
    checkbox_labels = ('postprocess', 'wiener', 'bilateral', 'laplacian')
    checkbox_values = (self.settings.use_postprocess, self.settings.use_wiener, 
                      self.settings.use_bilateral, self.settings.use_laplacian)
    
    # Create checkboxes horizontally using consistent helper
    checkbox_rect = layout_manager.add_component(0.06)
    self.checkboxes = create_checkboxes(checkbox_rect, checkbox_labels, checkbox_values)
    self.checkbox_labels = checkbox_labels
    
    # Define all sliders in one place
    slider_definitions = [
      ('gamma', 'gamma', 0.1, 3.0, nested('tonemap', 'gamma')),
      ('light', 'light_adapt', 0.0, 1.0, nested('tonemap', 'light_adapt')),
      ('intensity', 'intensity', -1.0, 5.0, nested('tonemap', 'intensity')),
      ('detail', 'bil_detail', 0.0, 2.0, field('bilateral_detail')),
      ('bil_sigma_s', 'bil_sigma_s', 1.0, 10.0, field('bilateral_sigma_s')),
      ('bil_sigma_r', 'bil_sigma_r', 0.01, 1.0, field('bilateral_sigma_r')),
      ('wiener', 'wiener_sigma', 0.001, 0.5, field('wiener_sigma')),
      ('vibrance', 'vibrance', -1.0, 1.0, field('vibrance')),
    ]
    
    # Create slider axes and sliders in one loop
    slider_axes = layout_manager.add_slider_group(len(slider_definitions))
    self.param_mappings = {}
    
    for i, (attr_name, label, min_val, max_val, mapping) in enumerate(slider_definitions):
      getter, _ = mapping
      slider = Slider(slider_axes[i], label, min_val, max_val, valinit=getter(self.settings))
      setattr(self, attr_name, slider)
      self.param_mappings[slider] = mapping
    
    # Connect event handlers
    self._connect_pipeline_events()
    
  def _connect_pipeline_events(self):
    """Connect all pipeline UI event handlers."""
    # Preset handler
    def on_presets(label):
      self.switch_preset(label)
      self._sync_ui_from_settings()
      self.update_display_callback()
      
    # Debayer handler
    def on_debayer(label):
      self.update_debayer_method(label)
      self.update_display_callback()
      
    # Tonemap handler  
    def on_tonemap(label):
      self.update_tonemap_method(label)
      self.update_display_callback()
      
    # Checkbox handler
    def on_checkbox(label):
      idx = self.checkbox_labels.index(label)
      is_checked = self.checkboxes[idx].get_status()[0]
      self.update_checkbox_setting(label, is_checked)
      self.update_display_callback()
      
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
    debayer_options = ('bilinear', 'rcd', 'ppg', 'opencv')
    try:
      debayer_index = debayer_options.index(self.settings.debayer)
    except ValueError:
      print(f"Warning: Unknown debayer method '{self.settings.debayer}', defaulting to 'rcd'")
      debayer_index = debayer_options.index('rcd')
    self.rb_debayer.set_active(debayer_index)
    
    tonemap_options = ('reinhard', 'aces', 'adaptive_aces', 'linear') 
    try:
      tonemap_index = tonemap_options.index(self.settings.tonemap_method)
    except ValueError:
      print(f"Warning: Unknown tonemap method '{self.settings.tonemap_method}', defaulting to 'reinhard'")
      tonemap_index = tonemap_options.index('reinhard')
    self.rb_tonemap.set_active(tonemap_index)
    
    # Update checkboxes
    checkbox_values = [self.settings.use_postprocess, self.settings.use_wiener, 
                      self.settings.use_bilateral, self.settings.use_laplacian]
    for cb, value in zip(self.checkboxes, checkbox_values, strict=True):
      cb.set_active(0, value)
