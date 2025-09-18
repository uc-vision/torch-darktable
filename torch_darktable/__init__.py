"""PyTorch CUDA extensions for image processing and computer vision."""

# Import all modules
from . import extension, tonemap, debayer, local_contrast, color_conversion, white_balance, denoise

from .tonemap import Reinhard, aces_tonemap, linear_tonemap, compute_image_bounds, compute_image_metrics, TonemapParameters
from .debayer import (
    BayerPattern, Packed12Format,
    create_ppg, create_rcd, create_postprocess,
    encode12, decode12,
    encode12_u16, encode12_float, decode12_float, decode12_half, decode12_u16,
    bilinear5x5_demosaic
)
from .local_contrast import LaplacianParams, create_laplacian, local_laplacian_rgb, create_bilateral, bilateral_rgb
from .color_conversion import (
    compute_luminance, modify_luminance,
    compute_log_luminance, modify_log_luminance,
    modify_saturation,
    rgb_to_lab, lab_to_rgb,
    rgb_to_xyz, xyz_to_lab, lab_to_xyz, xyz_to_rgb,
    color_transform_3x3
)
from .white_balance import apply_white_balance, estimate_white_balance
from .denoise import Wiener, create_wiener, estimate_channel_noise


__all__ = [
    # Core classes and enums
    "BayerPattern", "LaplacianParams", "Reinhard",
    
    # Tone mapping
    "aces_tonemap", "linear_tonemap", "compute_image_bounds", "compute_image_metrics", "TonemapParameters",
    
    # Debayering and 12-bit encoding
    "create_ppg", "create_rcd", "create_postprocess", "Packed12Format", "bilinear5x5_demosaic",
    "encode12", "decode12",
    "encode12_u16", "encode12_float", "decode12_float", "decode12_half", "decode12_u16",
    
    # Local contrast enhancement
    "create_laplacian", "local_laplacian_rgb",
    "create_bilateral", "bilateral_rgb",
    
    # Color conversions
    "compute_luminance", "modify_luminance",
    "compute_log_luminance", "modify_log_luminance",
    "modify_saturation", 
    "rgb_to_lab", "lab_to_rgb", 
    "rgb_to_xyz", "xyz_to_lab", "lab_to_xyz", "xyz_to_rgb",
    "color_transform_3x3",
    
    # White balance
    "apply_white_balance", "estimate_white_balance",
    
    # Wiener denoising
    "Wiener", "create_wiener", "estimate_channel_noise",


    # Submodules
    "extension",
    "tonemap",
    "debayer",
    "local_contrast",
    "color_conversion",
    "white_balance",
    "denoise",
]