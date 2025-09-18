import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import numpy as np
from PIL import Image
import torch

import torch_darktable as td

from .pipeline import ImagePipeline
from .util import CameraSettings, camera_settings, load_raw_image, settings_for_file


def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def parse_args():
  parser = argparse.ArgumentParser(description='Run inference on a single raw image using ZRR models')
  parser.add_argument('input', type=Path, help='Path to input raw image')
  parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')

  parser.add_argument(
    '--camera', type=str, default=None, help='Camera name (one of ' + ', '.join(camera_settings.keys()) + ')'
  )

  return parser.parse_args()


def interactive_debayer(bayer_image: torch.Tensor, input_path: Path, camera_settings: CameraSettings) -> None:
  device = bayer_image.device

  # Initialize settings with camera default preset
  settings = ImagePipeline.presets[camera_settings.preset]
  current_preset = camera_settings.preset

  # Create pipeline
  pipeline = ImagePipeline(device, camera_settings, settings)

  wb = td.estimate_white_balance([bayer_image], pattern=camera_settings.bayer_pattern, quantile=0.95)
  print('White balance: ', wb.tolist())

  def compute_rgb() -> np.ndarray:
    return pipeline.process(bayer_image, wb)

  rgb_np = compute_rgb()

  fig, ax = plt.subplots(figsize=(12, 8))
  plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.01)
  im = ax.imshow(rgb_np, interpolation='nearest')
  ax.set_aspect('equal', adjustable='box')
  ax.set_axis_off()

  # Left sidebar controls
  sidebar_x = 0.02
  sidebar_w = 0.20

  # Presets at top
  ax_presets = plt.axes((sidebar_x, 0.9, sidebar_w, 0.08))
  available_presets = list(ImagePipeline.presets.keys())
  rb_presets = RadioButtons(ax_presets, available_presets, active=available_presets.index(current_preset))

  # Debayer method
  ax_debayer = plt.axes((sidebar_x, 0.8, sidebar_w, 0.08))
  rb = RadioButtons(ax_debayer, ('bilinear', 'rcd', 'ppg'), active=('bilinear', 'rcd', 'ppg').index(settings.debayer))

  # Tonemap method
  ax_tonemap = plt.axes((sidebar_x, 0.7, sidebar_w, 0.08))
  rb_tm = RadioButtons(
    ax_tonemap,
    ('reinhard', 'aces', 'linear'),
    active=('reinhard', 'aces', 'linear').index(settings.tonemap_method),
  )

  # Checkboxes
  checkbox_labels = ('postprocess', 'wiener', 'bilateral', 'white_balance')
  ax_checks = plt.axes((sidebar_x, 0.6, sidebar_w, 0.08))
  cb = CheckButtons(
    ax_checks,
    checkbox_labels,
    (settings.use_postprocess, settings.use_wiener, settings.use_bilateral, settings.use_white_balance),
  )
  # Sliders in sidebar - tightly packed

  def create_axes_vertical(
    n: int,
    x: float = sidebar_x + 0.06,
    w: float = sidebar_w - 0.07,
    h: float = 0.015,
    y_top: float = 0.55,
    y_bottom: float = 0.35,
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
  gamma = Slider(ax_gamma, 'gamma', 0.1, 3.0, valinit=settings.tonemap.gamma)
  light = Slider(ax_light, 'light_adapt', 0.0, 1.0, valinit=settings.tonemap.light_adapt)

  # Bilateral group
  detail = Slider(ax_detail, 'bil_detail', 0.0, 2.0, valinit=settings.bilateral_detail)

  # Wiener group
  wiener = Slider(ax_wiener_sigma, 'wiener_sigma', 0.001, 0.5, valinit=settings.wiener_sigma)

  # Color adjustment group
  vibrance = Slider(ax_vibrance, 'vibrance', -1.0, 1.0, valinit=settings.vibrance)
  intensity = Slider(ax_intensity, 'intensity', -1.0, 3.0, valinit=settings.tonemap.intensity)

  # Save button at bottom of sidebar
  ax_save = plt.axes((sidebar_x, 0.02, sidebar_w, 0.06))
  btn_save = Button(ax_save, 'Save JPEG')

  ax_detail.set_zorder(10)
  ax_wiener_sigma.set_zorder(10)

  # Parameter handling abstraction
  def field(name):
    return (lambda s: getattr(s, name), lambda s, val: replace(s, **{name: val}))

  def nested(outer, inner):
    return (
      lambda s: getattr(getattr(s, outer), inner),
      lambda s, val: replace(s, **{outer: replace(getattr(s, outer), **{inner: val})}),
    )

  def make_param_handler(getter, setter):
    def handler(val):
      nonlocal settings
      settings = setter(settings, float(val))
      update_display()

    return handler

  def sync_ui_from_settings(settings_obj):
    for slider, (getter, _) in param_mappings.items():
      slider.set_val(getter(settings_obj))

    # Update radio buttons
    rb.set_active(('bilinear', 'rcd', 'ppg').index(settings_obj.debayer))
    rb_tm.set_active(('reinhard', 'aces', 'linear').index(settings_obj.tonemap_method))

    # Update checkboxes
    cb.set_active(0, settings_obj.use_postprocess)
    cb.set_active(1, settings_obj.use_wiener)
    cb.set_active(2, settings_obj.use_bilateral)
    cb.set_active(3, settings_obj.use_white_balance)

  # Parameter mappings
  param_mappings = {
    gamma: nested('tonemap', 'gamma'),
    light: nested('tonemap', 'light_adapt'),
    intensity: nested('tonemap', 'intensity'),
    detail: field('bilateral_detail'),
    wiener: field('wiener_sigma'),
    vibrance: field('vibrance'),
  }

  def update_display(**kwargs):
    nonlocal pipeline, settings
    if kwargs:
      settings = replace(settings, **kwargs)
    pipeline = ImagePipeline(device, camera_settings, settings)
    new_img = compute_rgb()
    im.set_data(new_img)
    fig.canvas.draw_idle()

  def on_presets(label):
    nonlocal settings, current_preset
    current_preset = label
    settings = ImagePipeline.presets[label]

    sync_ui_from_settings(settings)

    update_display()

  def on_rb(label):
    update_display(debayer=label)  # type: ignore[arg-type]

  def on_cb(label):
    # Get current checkbox state (after the click)
    idx = checkbox_labels.index(label)
    is_checked = cb.get_status()[idx]

    if label == 'postprocess':
      update_display(use_postprocess=is_checked)
    elif label == 'wiener':
      update_display(use_wiener=is_checked)
    elif label == 'bilateral':
      update_display(use_bilateral=is_checked)
    elif label == 'white_balance':
      update_display(use_white_balance=is_checked)

  def on_rb_tm(label):
    update_display(tonemap_method=label)  # type: ignore[arg-type]

  def on_save_jpeg(event):
    # Get current processed image
    rgb_array = compute_rgb()

    # Convert to PIL Image and save as JPEG
    if rgb_array.dtype != np.uint8:
      rgb_array = (rgb_array * 255).astype(np.uint8)

    pil_image = Image.fromarray(rgb_array)

    # Create JPEG filename from input path
    output_path = input_path.with_suffix('.jpg')
    pil_image.save(output_path, 'JPEG', quality=95)
    print(f'Saved JPEG to: {output_path}')

  # Auto-register parameter handlers
  for slider, (getter, setter) in param_mappings.items():
    slider.on_changed(make_param_handler(getter, setter))

  rb_presets.on_clicked(on_presets)
  rb.on_clicked(on_rb)
  rb_tm.on_clicked(on_rb_tm)
  cb.on_clicked(on_cb)
  btn_save.on_clicked(on_save_jpeg)

  plt.show()


def main():
  args = parse_args()
  set_seed(args.seed)
  assert args.input.exists(), f'Error: Input file {args.input} does not exist'

  # Get camera settings explicitly
  if args.camera:
    if args.camera not in camera_settings:
      raise ValueError(f'Unknown camera: {args.camera}. Available cameras: {list(camera_settings.keys())}')
    cam_settings = camera_settings[args.camera]
  else:
    cam_settings = settings_for_file(args.input)

  bayer_image = load_raw_image(args.input, cam_settings)
  interactive_debayer(bayer_image, args.input, cam_settings)


if __name__ == '__main__':
  main()
