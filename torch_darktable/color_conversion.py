"""Color space conversion functions."""

import torch
from .extension import extension


def compute_luminance(rgb_image: torch.Tensor) -> torch.Tensor:
    """
    Compute luminance from RGB image.
    
    Args:
        rgb_image: Input RGB image tensor
        
    Returns:
        Luminance tensor
    """
    return extension.compute_luminance(rgb_image)


def modify_luminance(
    rgb_image: torch.Tensor,
    luminance_multiplier: torch.Tensor
) -> torch.Tensor:
    """
    Modify luminance of RGB image while preserving chromaticity.
    
    Args:
        rgb_image: Input RGB image tensor
        luminance_multiplier: Luminance scaling factor tensor
        
    Returns:
        Modified RGB image tensor
    """
    return extension.modify_luminance(rgb_image, luminance_multiplier)


def rgb_to_lab(rgb_image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to CIELAB color space.
    
    Args:
        rgb_image: Input RGB image tensor
        
    Returns:
        LAB image tensor
    """
    return extension.rgb_to_lab(rgb_image)


def lab_to_rgb(lab_image: torch.Tensor) -> torch.Tensor:
    """
    Convert CIELAB to RGB color space.
    
    Args:
        lab_image: Input LAB image tensor
        
    Returns:
        RGB image tensor
    """
    return extension.lab_to_rgb(lab_image)


def rgb_to_xyz(rgb_image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to CIE XYZ color space.
    
    Args:
        rgb_image: Input RGB image tensor
        
    Returns:
        XYZ image tensor
    """
    return extension.rgb_to_xyz(rgb_image)


def xyz_to_lab(xyz_image: torch.Tensor) -> torch.Tensor:
    """
    Convert CIE XYZ to CIELAB color space.
    
    Args:
        xyz_image: Input XYZ image tensor
        
    Returns:
        LAB image tensor
    """
    return extension.xyz_to_lab(xyz_image)


def lab_to_xyz(lab_image: torch.Tensor) -> torch.Tensor:
    """
    Convert CIELAB to CIE XYZ color space.
    
    Args:
        lab_image: Input LAB image tensor
        
    Returns:
        XYZ image tensor
    """
    return extension.lab_to_xyz(lab_image)


def xyz_to_rgb(xyz_image: torch.Tensor) -> torch.Tensor:
    """
    Convert CIE XYZ to RGB color space.
    
    Args:
        xyz_image: Input XYZ image tensor
        
    Returns:
        RGB image tensor
    """
    return extension.xyz_to_rgb(xyz_image)


__all__ = [
    "compute_luminance", "modify_luminance",
    "rgb_to_lab", "lab_to_rgb",
    "rgb_to_xyz", "xyz_to_lab", "lab_to_xyz", "xyz_to_rgb"
]
