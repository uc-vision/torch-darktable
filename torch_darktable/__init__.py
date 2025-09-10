"""PyTorch CUDA extensions for image processing and computer vision."""

# Import all modules
from . import extension
from .tonemap import Reinhard, aces_tonemap, compute_image_bounds
from .debayer import (
    BayerPattern, Packed12Format,
    create_ppg, create_rcd, create_postprocess,
    encode12, decode12,
    encode12_u16, encode12_float, decode12_float, decode12_half, decode12_u16,
)
from .local_contrast import LaplacianParams, create_laplacian, local_laplacian_rgb, create_bilateral, bilateral_rgb
from .color_conversion import (
    compute_luminance, modify_luminance,
    rgb_to_lab, lab_to_rgb,
    rgb_to_xyz, xyz_to_lab, lab_to_xyz, xyz_to_rgb
)

__all__ = [
    # Core classes and enums
    "BayerPattern", "LaplacianParams", "Reinhard",
    
    # Tone mapping
    "aces_tonemap", "compute_image_bounds",
    
    # Debayering and 12-bit encoding
    "create_ppg", "create_rcd", "create_postprocess", "Packed12Format",
    "encode12", "decode12",
    "encode12_u16", "encode12_float", "decode12_float", "decode12_half", "decode12_u16",
    
    # Local contrast enhancement
    "create_laplacian", "local_laplacian_rgb",
    "create_bilateral", "bilateral_rgb",
    
    # Color conversions
    "compute_luminance", "modify_luminance",
    "rgb_to_lab", "lab_to_rgb", 
    "rgb_to_xyz", "xyz_to_lab", "lab_to_xyz", "xyz_to_rgb",

    # Extension
    "extension"
]