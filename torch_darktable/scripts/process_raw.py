from pathlib import Path
import torch
import os
import numpy as np
import argparse
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from PIL import Image

import torch_darktable as td
from torch_darktable.tonemap import TonemapParameters
from .util import load_raw_image, camera_settings

from beartype import beartype


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on a single raw image using ZRR models')
    parser.add_argument('input', type=Path, help='Path to input raw image')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')

    parser.add_argument('--camera', type=str, default=None, help='Camera name (one of ' + ', '.join(camera_settings.keys()) + ')')

    return parser.parse_args()

@beartype
@dataclass
class Settings:
    debayer: Literal['bilinear', 'rcd', 'ppg'] = 'rcd'
    tonemap_method: Literal['reinhard', 'aces', 'linear'] = 'reinhard'
    use_postprocess: bool = True
    use_bilateral: bool = True
    use_wiener: bool = False
    
    bilateral_detail: float = 0.2
    bilateral_sigma_s: float = 4.0
    bilateral_sigma_r: float = 0.2
    ppg_median_threshold: float = 0.0
    postprocess_green_eq_threshold: float = 0.04

    tonemap: TonemapParameters = field(default_factory=lambda: TonemapParameters(
        gamma=0.75,
        light_adapt=0.9,
        intensity=1.0
    ))

    wiener_sigma: float = 0.1
    saturation: float = 0.0


def interactive_debayer(bayer_image: torch.Tensor, input_path: Path) -> None:
    device = bayer_image.device
    image_size = (bayer_image.shape[1], bayer_image.shape[0])

    settings = Settings()


    # Create bilateral filter
    bil_workspace = td.create_bilateral(
        device,
        image_size,
        sigma_s=settings.bilateral_sigma_s,
        sigma_r=settings.bilateral_sigma_r
    )

    rcd_workspace = td.create_rcd(device, image_size, td.BayerPattern.RGGB, input_scale=1.0, output_scale=1.0)
    ppg_workspace = td.create_ppg(device, image_size, td.BayerPattern.RGGB, median_threshold=0.25)
    postprocess_workspace = td.create_postprocess(device, image_size, td.BayerPattern.RGGB, color_smoothing_passes=5, green_eq_local=True, green_eq_global=True, green_eq_threshold=settings.postprocess_green_eq_threshold)
    wiener_workspace = td.create_wiener(device, image_size)

    def compute_rgb() -> np.ndarray:
        # Use torch_darktable linear debayer
        if settings.debayer == 'bilinear':
            rgb_raw = td.bilinear5x5_demosaic(bayer_image.unsqueeze(-1), td.BayerPattern.RGGB)
        elif settings.debayer == 'rcd':
            rgb_raw = rcd_workspace.process(bayer_image.unsqueeze(-1))
        elif settings.debayer == 'ppg':
            rgb_raw = ppg_workspace.process(bayer_image.unsqueeze(-1))
        else:
            assert False, f"Invalid debayer method: {settings.debayer}"

        if settings.use_postprocess:
            rgb_raw = postprocess_workspace.process(rgb_raw)

        if settings.use_wiener:
            rgb_raw = wiener_workspace.process_log(rgb_raw, settings.wiener_sigma)
            

        if settings.use_bilateral:
            rgb_raw = td.bilateral_rgb(
                bil_workspace,
                rgb_raw,
                detail=settings.bilateral_detail,
            )

        # Apply saturation adjustment before tonemapping (if needed)  
        if settings.saturation != 0.0:
            rgb_raw = td.modify_saturation_mult_add(rgb_raw, 1.0, settings.saturation)

        # Compute metrics once
        metrics = td.Reinhard.compute_metrics(rgb_raw, stride=4, min_gray=1e-2)
        params = settings.tonemap

        if settings.tonemap_method == 'reinhard':
            rgb_tm = td.Reinhard.tonemap(
                rgb_raw, metrics, params)
        elif settings.tonemap_method == 'linear':
            rgb_tm = td.linear_tonemap(
                rgb_raw, metrics, params)
        elif settings.tonemap_method == 'aces':
            rgb_tm = td.aces_tonemap(rgb_raw, metrics, params)

        else:
          assert False, f"Invalid tonemap method: {settings.tonemap_method}"

        return rgb_tm.cpu().numpy()

    rgb_np = compute_rgb()

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.30)
    im = ax.imshow(rgb_np, interpolation='nearest')
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    # Left column: radio + compact checkboxes
    ax_debayer = plt.axes((0.05, 0.205, 0.14, 0.1))
    rb = RadioButtons(ax_debayer, ('bilinear', 'rcd', 'ppg'), active=('bilinear', 'rcd', 'ppg').index(settings.debayer))


    ax_checks = plt.axes((0.05, 0.09, 0.14, 0.1))
    cb = CheckButtons(
        ax_checks,
        ('postprocess', 'wiener', 'bilateral'),
        (settings.use_postprocess, settings.use_wiener, settings.use_bilateral)
    )

    # Tonemap method radio
    ax_tonemap = plt.axes((0.05, 0.015, 0.14, 0.06))
    rb_tm = RadioButtons(
        ax_tonemap,
        ('reinhard', 'aces', 'linear'),
        active=('reinhard', 'aces', 'linear').index(settings.tonemap_method),
    )

    # Right column: create axes first, then sliders
    def create_axes_vertical(n:int, x:float=0.30, w:float=0.65, h:float=0.03,
                             y_top:float=0.265, y_bottom:float=0.055):
        axes = []
        if n <= 0:
            return axes
        for i in range(n):
            t = (i / (n - 1)) if n > 1 else 0.0
            y = y_top - (y_top - y_bottom) * t
            axes.append(plt.axes((x, y, w, h)))
        return axes

    ax_gamma, ax_light, ax_detail, ax_sigma, ax_sigma_r, ax_wiener_sigma, ax_ppg_med, ax_postprocess_thresh, ax_intensity, ax_saturation = create_axes_vertical(10)

    # Tonemap group
    s_gamma = Slider(ax_gamma, 'gamma', 0.1, 3.0, valinit=settings.tonemap.gamma)
    s_light = Slider(ax_light, 'light_adapt', 0.0, 1.0, valinit=settings.tonemap.light_adapt)
    
    # Bilateral group
    s_detail = Slider(ax_detail, 'bil_detail', 0.0, 2.0, valinit=settings.bilateral_detail)
    s_sigma = Slider(ax_sigma, 'bil_sigma_s', 1.0, 16.0, valinit=settings.bilateral_sigma_s)
    s_sigma_r = Slider(ax_sigma_r, 'bil_sigma_r', 0.01, 0.5, valinit=settings.bilateral_sigma_r)
    
    # Wiener group
    s_wiener_sigma = Slider(ax_wiener_sigma, 'wiener_sigma', 0.001, 0.5, valinit=settings.wiener_sigma)
    
    # PPG group
    s_ppg_med = Slider(ax_ppg_med, 'ppg_median', 0.0, 1.0, valinit=settings.ppg_median_threshold)
    
    # Postprocess group
    s_postprocess_thresh = Slider(ax_postprocess_thresh, 'postproc_thresh', 0.001, 0.1, valinit=settings.postprocess_green_eq_threshold)
    
    # Color adjustment group
    s_saturation = Slider(ax_saturation, 'saturation', -1.0, 1.0, valinit=settings.saturation)
    s_intensity = Slider(ax_intensity, 'intensity', -1.0, 3.0, valinit=settings.tonemap.intensity)
    
    # Save button
    ax_save = plt.axes((0.22, 0.015, 0.06, 0.04))
    btn_save = Button(ax_save, 'Save JPEG')

    ax_detail.set_zorder(10)
    ax_sigma.set_zorder(10)
    ax_sigma_r.set_zorder(10)
    ax_wiener_sigma.set_zorder(10)
    ax_postprocess_thresh.set_zorder(10)

    def update_display():
        new_img = compute_rgb()
        im.set_data(new_img)
        fig.canvas.draw_idle()
      
    def on_rb(label):
        settings.debayer = label  # type: ignore[assignment]
        set_ppg_enabled(settings.debayer == 'ppg')
        update_display()

    def set_detail_enabled(enabled: bool):
        s_detail.set_active(enabled)
        ax_detail.patch.set_alpha(1.0 if enabled else 0.3)

    def set_wiener_enabled(enabled: bool):
        s_wiener_sigma.set_active(enabled)
        ax_wiener_sigma.patch.set_alpha(1.0 if enabled else 0.3)

    def update_sigma_enabled():
        enabled = settings.use_bilateral
        s_sigma.set_active(enabled)
        s_sigma_r.set_active(enabled)
        ax_sigma.patch.set_alpha(1.0 if enabled else 0.3)
        ax_sigma_r.patch.set_alpha(1.0 if enabled else 0.3)

    def set_bilateral_enabled(enabled: bool):
        set_detail_enabled(enabled)
        update_sigma_enabled()

    def on_cb(label):
        if label == 'postprocess':
            settings.use_postprocess = not settings.use_postprocess
            set_postprocess_enabled(settings.use_postprocess)
        elif label == 'wiener':
            settings.use_wiener = not settings.use_wiener
            set_wiener_enabled(settings.use_wiener)
        elif label == 'bilateral':
            settings.use_bilateral = not settings.use_bilateral
            set_bilateral_enabled(settings.use_bilateral)
        update_display()

    def on_rb_tm(label):
        settings.tonemap_method = label  # type: ignore[assignment]
        update_display()

    def on_gamma(val):
        settings.tonemap.gamma = float(val)
        update_display()

    def on_saturation(val):
        settings.saturation = float(val)
        update_display()
    
    def on_intensity(val):
        settings.tonemap.intensity = float(val)
        update_display()

    def on_detail(val):
        settings.bilateral_detail = float(val)
        update_display()

    def on_wiener_sigma(val):
        settings.wiener_sigma = float(val)
        update_display()

    def on_sigma(val):
        settings.bilateral_sigma_s = float(val)
        update_display()

    def on_light(val):
        settings.tonemap.light_adapt = float(val)
        update_display()

    def on_sigma_r(val):
        settings.bilateral_sigma_r = float(val)
        update_display()

    def set_ppg_enabled(enabled: bool):
        s_ppg_med.set_active(enabled)
        ax_ppg_med.patch.set_alpha(1.0 if enabled else 0.3)

    def set_postprocess_enabled(enabled: bool):
        s_postprocess_thresh.set_active(enabled)
        ax_postprocess_thresh.patch.set_alpha(1.0 if enabled else 0.3)

    def on_ppg_median(val):
        nonlocal ppg_workspace
        settings.ppg_median_threshold = float(val)
        ppg_workspace.median_threshold = settings.ppg_median_threshold
        update_display()
    
    def on_postprocess_thresh(val):
        nonlocal postprocess_workspace
        settings.postprocess_green_eq_threshold = float(val)
        postprocess_workspace.green_eq_threshold = settings.postprocess_green_eq_threshold
        update_display()
    
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
        print(f"Saved JPEG to: {output_path}")

    rb.on_clicked(on_rb)
    rb_tm.on_clicked(on_rb_tm)
    cb.on_clicked(on_cb)
    s_gamma.on_changed(on_gamma)
    s_detail.on_changed(on_detail)
    s_wiener_sigma.on_changed(on_wiener_sigma)
    s_sigma.on_changed(on_sigma)
    s_light.on_changed(on_light)
    s_sigma_r.on_changed(on_sigma_r)
    s_ppg_med.on_changed(on_ppg_median)
    s_postprocess_thresh.on_changed(on_postprocess_thresh)
    s_saturation.on_changed(on_saturation)
    s_intensity.on_changed(on_intensity)
    btn_save.on_clicked(on_save_jpeg)

    set_bilateral_enabled(settings.use_bilateral)
    set_wiener_enabled(settings.use_wiener)
    set_postprocess_enabled(settings.use_postprocess)
    update_sigma_enabled()
    set_ppg_enabled(settings.debayer == 'ppg')

    plt.show()


def main():
    args = parse_args()
    set_seed(args.seed)
    assert os.path.exists(args.input), f"Error: Input file {args.input} does not exist"

    bayer_image = load_raw_image(args.input, args.camera) 
    interactive_debayer(bayer_image, args.input)


if __name__ == "__main__":
    main()