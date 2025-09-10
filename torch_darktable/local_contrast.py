"""Local contrast enhancement algorithms."""

import torch
from dataclasses import dataclass
from .extension import extension


@dataclass
class LaplacianParams:
    """Parameters for local Laplacian filtering."""
    num_gamma: int = 6
    sigma: float = 0.2
    shadows: float = 1.0
    highlights: float = 1.0
    clarity: float = 0.0


def create_laplacian(
    device: torch.device,
    image_size: tuple[int, int],
    params: LaplacianParams
) -> "extension.Laplacian":
    """
    Create a local Laplacian filter object.
    
    Args:
        device: CUDA device to use
        image_size: (width, height) of the image
        params: Laplacian filter parameters
        
    Returns:
        Laplacian algorithm object
    """
    width, height = image_size
    # Map params to implementation defaults conservatively
    return extension.Laplacian(
        device,
        width,
        height,
        params.num_gamma,
        params.sigma,
        params.shadows,
        params.highlights,
        params.clarity,
    )


def local_laplacian_rgb(
    laplacian: "extension.Laplacian",
    input_image: torch.Tensor,
) -> torch.Tensor:
    """
    Apply local Laplacian filtering to RGB image.
    
    Args:
        laplacian: Laplacian algorithm object
        input_image: Input RGB image tensor
        
    Returns:
        Filtered RGB image tensor
    """
    return laplacian.process(input_image)


def create_bilateral(
    device: torch.device,
    image_size: tuple[int, int],
    spatial_sigma: float,
    range_sigma: float,
    detail: float = 0.0
) -> "extension.Bilateral":
    """
    Create a bilateral filter object.
    
    Args:
        device: CUDA device to use
        image_size: (width, height) of the image
        spatial_sigma: Spatial standard deviation
        range_sigma: Range standard deviation
        
    Returns:
        Bilateral algorithm object
    """
    width, height = image_size
    return extension.Bilateral(device, width, height, spatial_sigma, range_sigma, detail)


def bilateral_rgb(
    bilateral: "extension.Bilateral",
    input_image: torch.Tensor,
) -> torch.Tensor:
    """
    Apply bilateral filtering to RGB image.
    
    Args:
        bilateral: Bilateral algorithm object
        input_image: Input RGB image tensor
        
    Returns:
        Filtered RGB image tensor
    """
    luminance = extension.compute_luminance(input_image)
    filtered_lum = bilateral.process(luminance)
    return extension.modify_luminance(input_image, filtered_lum)


__all__ = [
    "LaplacianParams", 
    "create_laplacian", "local_laplacian_rgb",
    "create_bilateral", "bilateral_rgb"
]
