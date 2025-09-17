"""Local contrast enhancement algorithms."""

import torch
from dataclasses import dataclass
from .extension import extension
from beartype import beartype


@beartype
@dataclass
class LaplacianParams:
    """Parameters for local Laplacian filtering."""
    num_gamma: int = 6
    sigma: float = 0.2
    shadows: float = 1.0
    highlights: float = 1.0
    clarity: float = 0.0


@beartype
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


@beartype
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
    luminance = extension.compute_luminance(input_image)
    return extension.modify_luminance(input_image, laplacian.process(luminance))


@beartype
def create_bilateral(
    device: torch.device,
    image_size: tuple[int, int],
    *,
    sigma_s: float,
    sigma_r: float,
) -> "extension.Bilateral":
    """
    Create a bilateral filter object.
    
    Args:
        device: CUDA device to use
        image_size: (width, height) of the image
        sigma_s: Spatial standard deviation
        sigma_r: Luminance range standard deviation (determines subdivisions)
        
    Returns:
        Bilateral algorithm object
    """
    width, height = image_size
    return extension.Bilateral(device, width, height, sigma_s, sigma_r)


@beartype
def bilateral_rgb(
    bilateral: "extension.Bilateral",
    input_image: torch.Tensor,
    detail: float
) -> torch.Tensor:
    luminance = extension.compute_luminance(input_image)
    return extension.modify_luminance(input_image, bilateral.process(luminance, float(detail)))

@beartype
def log_bilateral_rgb(
    bilateral: "extension.Bilateral",
    input_image: torch.Tensor,
    detail: float,
    eps: float = 1e-6
) -> torch.Tensor:
    log_luminance = extension.compute_log_luminance(input_image, eps)
    return extension.modify_log_luminance(input_image, bilateral.process(log_luminance, float(detail)), eps)



__all__ = [
    "LaplacianParams", 
    "create_laplacian", "local_laplacian_rgb",
    "create_bilateral", "bilateral_rgb",
]
