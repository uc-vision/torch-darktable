from pathlib import Path
import torch
import os
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

from image_isp.load import add_camera_settings, load_raw_image, settings_from_args, stack_bayer
import torch_darktable as td


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on a single raw image using ZRR models')
    parser.add_argument('input', type=Path, help='Path to input raw image')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')

    add_camera_settings(parser)

    return parser.parse_args()


@dataclass
class Settings:
    debayer: Literal['linear', 'rcd', 'ppg'] = 'rcd'
    tonemap_method: Literal['reinhard', 'aces', 'linear'] = 'reinhard'
    use_postprocess: bool = True
    use_bilateral: bool = True
    use_bilateral_denoise: bool = False
    denoise_amount: float = 0.5
    
    bilateral_detail: float = 0.2
    bilateral_sigma_s: float = 2.0
    bilateral_sigma_r: float = 0.2
    ppg_median_threshold: float = 0.25
    tonemap_gamma: float = 0.75
    tonemap_intensity: float = 2.0
    tonemap_light_adapt: float = 0.9


def interactive_debayer(bayer_image: torch.Tensor) -> None:
    device = bayer_image.device
    image_size = (bayer_image.shape[1], bayer_image.shape[0])

    settings = Settings()


    # Create bilateral filter
    bil_workspace = td.create_bilateral(
        device,
        image_size,
        spatial_sigma=settings.bilateral_sigma_s,
        range_sigma=settings.bilateral_sigma_r
    )

    rcd_workspace = td.create_rcd(device, image_size, td.BayerPattern.RGGB, input_scale=1.0, output_scale=1.0)
    ppg_workspace = td.create_ppg(device, image_size, td.BayerPattern.RGGB, median_threshold=0.25)
    postprocess_workspace = td.create_postprocess(device, image_size, td.BayerPattern.RGGB, color_smoothing_passes=1, green_eq_local=True, green_eq_global=True, green_eq_threshold=0.01)

    def compute_rgb() -> np.ndarray:
        # Use torch_darktable linear debayer
        if settings.debayer == 'linear':
            rgb_raw = td.bilinear5x5_demosaic(bayer_image.unsqueeze(-1), td.BayerPattern.RGGB)
        elif settings.debayer == 'rcd':
            rgb_raw = rcd_workspace.process(bayer_image.unsqueeze(-1))
        else:  # PPG
            rgb_raw = ppg_workspace.process(bayer_image.unsqueeze(-1))

        if settings.use_postprocess and settings.debayer != 'linear':
            rgb_raw = postprocess_workspace.process(rgb_raw)

        if settings.use_bilateral_denoise or settings.use_bilateral:
            rgb_raw = td.bilateral_rgb(
                bil_workspace,
                rgb_raw,
                detail=(settings.bilateral_detail if settings.use_bilateral else None),
                denoise_amount=(settings.denoise_amount if settings.use_bilateral_denoise else None),
            )

        if settings.tonemap_method == 'reinhard':
            # Compute metrics and apply Reinhard tone mapping
            metrics = td.Reinhard.compute_metrics(rgb_raw, stride=4, min_gray=1e-2)
            rgb_tm = td.Reinhard.tonemap(
                rgb_raw,
                metrics,
                gamma=settings.tonemap_gamma,
                intensity=settings.tonemap_intensity,
                light_adapt=settings.tonemap_light_adapt,
            )
        elif settings.tonemap_method == 'linear':
            max_val = rgb_raw.max()
            rgb_tm = (rgb_raw / (max_val + 1e-8)) * (2 ** (settings.tonemap_intensity))
            rgb_tm = rgb_tm ** (1.0 / settings.tonemap_gamma)
            rgb_tm = (rgb_tm.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        elif settings.tonemap_method == 'aces':
            exposure_adjusted = rgb_raw.clamp(0.0, 1.0) * (2 ** (settings.tonemap_intensity + 2))
            rgb_tm = td.aces_tonemap(exposure_adjusted, gamma=settings.tonemap_gamma)
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
    rb = RadioButtons(ax_debayer, ('linear', 'rcd', 'ppg'), active=('linear', 'rcd', 'ppg').index(settings.debayer))


    ax_checks = plt.axes((0.05, 0.09, 0.14, 0.1))
    cb = CheckButtons(
        ax_checks,
        ('postprocess', 'bilat_denoise', 'bilateral'),
        (settings.use_postprocess, settings.use_bilateral_denoise, settings.use_bilateral)
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
                             y_top:float=0.265, y_bottom:float=0.055) -> list[plt.Axes]:
        axes: list[plt.Axes] = []
        if n <= 0:
            return axes
        for i in range(n):
            t = (i / (n - 1)) if n > 1 else 0.0
            y = y_top - (y_top - y_bottom) * t
            axes.append(plt.axes((x, y, w, h)))
        return axes

    ax_gamma, ax_intensity, ax_denoise, ax_detail, ax_sigma, ax_sigma_r, ax_light, ax_ppg_med = create_axes_vertical(8)

    s_gamma = Slider(ax_gamma, 'gamma', 0.1, 3.0, valinit=settings.tonemap_gamma)
    s_intensity = Slider(ax_intensity, 'intensity', 0.1, 10.0, valinit=settings.tonemap_intensity)
    s_denoise = Slider(ax_denoise, 'bilat_denoise', 0.0, 1.0, valinit=settings.denoise_amount)
    s_detail = Slider(ax_detail, 'bilat_detail', 0.0, 2.0, valinit=settings.bilateral_detail)
    s_sigma = Slider(ax_sigma, 'bilat_sigma', 1.0, 16.0, valinit=settings.bilateral_sigma_s)
    s_sigma_r = Slider(ax_sigma_r, 'bilat_sigma_r', 0.01, 0.5, valinit=settings.bilateral_sigma_r)
    s_light = Slider(ax_light, 'light_adapt', 0.0, 1.0, valinit=settings.tonemap_light_adapt)
    s_ppg_med = Slider(ax_ppg_med, 'ppg_median', 0.0, 1.0, valinit=settings.ppg_median_threshold)

    ax_detail.set_zorder(10)
    ax_sigma.set_zorder(10)
    ax_sigma_r.set_zorder(10)

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

    def set_denoise_enabled(enabled: bool):
        s_denoise.set_active(enabled)
        ax_denoise.patch.set_alpha(1.0 if enabled else 0.3)

    def update_sigma_enabled():
        enabled = settings.use_bilateral or settings.use_bilateral_denoise
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
        elif label == 'bilat_denoise':
            settings.use_bilateral_denoise = not settings.use_bilateral_denoise
            set_denoise_enabled(settings.use_bilateral_denoise)
            update_sigma_enabled()
        elif label == 'bilateral':
            settings.use_bilateral = not settings.use_bilateral
            set_bilateral_enabled(settings.use_bilateral)
        update_display()

    def on_rb_tm(label):
        settings.tonemap_method = label  # type: ignore[assignment]
        update_display()

    def on_gamma(val):
        settings.tonemap_gamma = float(val)
        update_display()

    def on_intensity(val):
        settings.tonemap_intensity = float(val)
        update_display()

    def on_detail(val):
        settings.bilateral_detail = float(val)
        update_display()

    def on_denoise(val):
        settings.denoise_amount = float(val)
        update_display()

    def on_sigma(val):
        settings.bilateral_sigma_s = float(val)
        update_display()

    def on_light(val):
        settings.tonemap_light_adapt = float(val)
        update_display()

    def on_sigma_r(val):
        settings.bilateral_sigma_r = float(val)
        update_display()

    def set_ppg_enabled(enabled: bool):
        s_ppg_med.set_active(enabled)
        ax_ppg_med.patch.set_alpha(1.0 if enabled else 0.3)

    def on_ppg_median(val):
        nonlocal ppg_workspace
        settings.ppg_median_threshold = float(val)
        ppg_workspace.median_threshold = settings.ppg_median_threshold
        update_display()

    rb.on_clicked(on_rb)
    rb_tm.on_clicked(on_rb_tm)
    cb.on_clicked(on_cb)
    s_gamma.on_changed(on_gamma)
    s_intensity.on_changed(on_intensity)
    s_detail.on_changed(on_detail)
    s_denoise.on_changed(on_denoise)
    s_sigma.on_changed(on_sigma)
    s_light.on_changed(on_light)
    s_sigma_r.on_changed(on_sigma_r)
    s_ppg_med.on_changed(on_ppg_median)

    set_bilateral_enabled(settings.use_bilateral)
    set_denoise_enabled(settings.use_bilateral_denoise)
    update_sigma_enabled()
    set_ppg_enabled(settings.debayer == 'ppg')

    plt.show()


def main():
    args = parse_args()
    set_seed(args.seed)
    assert os.path.exists(args.input), f"Error: Input file {args.input} does not exist"
    camera_settings = settings_from_args(args)

    bayer_image = load_raw_image(args.input, camera_settings) * camera_settings.brightness
    interactive_debayer(bayer_image)


if __name__ == "__main__":
    main()