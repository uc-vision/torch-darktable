

import torch
from enum import Enum
from pathlib import Path
from torch.utils import cpp_extension
from dataclasses import dataclass


class BayerPattern(Enum):
  RGGB = 0x94949494
  BGGR = 0x16161616
  GRBG = 0x61616161
  GBRG = 0x49494949


@dataclass
class LaplacianParams:
    """Parameters for local Laplacian filtering."""
    
    num_gamma: int = 6
    """Number of gamma levels for tone mapping (4, 6, or 8 supported)"""
    
    sigma: float = 0.2
    """Tone mapping parameter controlling transitions between regions (0.0-1.0)"""
    
    shadows: float = 0.0
    """Shadow enhancement (-1.0 to 1.0, negative lifts shadows)"""
    
    highlights: float = 0.0
    """Highlight compression (-1.0 to 1.0, positive compresses highlights)"""
    
    clarity: float = 0.0
    """Local contrast enhancement (-1.0 to 1.0, positive increases clarity)"""
    
    def __post_init__(self):
        """Validate parameter ranges."""
        if self.num_gamma not in [4, 6, 8]:
            raise ValueError(f"Gamma levels must be 4, 6, or 8, got {self.num_gamma}")
        if not 0.0 <= self.sigma <= 1.0:
            raise ValueError(f"Sigma must be in range [0.0, 1.0], got {self.sigma}")
        if not -1.0 <= self.shadows <= 1.0:
            raise ValueError(f"Shadows must be in range [-1.0, 1.0], got {self.shadows}")
        if not -1.0 <= self.highlights <= 1.0:
            raise ValueError(f"Highlights must be in range [-1.0, 1.0], got {self.highlights}")
        if not -1.0 <= self.clarity <= 1.0:
            raise ValueError(f"Clarity must be in range [-1.0, 1.0], got {self.clarity}")


# Dynamic compilation of CUDA extension
def _load_cuda_extension(debug=False):
    """Load the CUDA extension dynamically"""
    current_dir = Path(__file__).parent
    source_dir = current_dir / "csrc"
    
    sources = [
        str(source_dir / "extension.cpp"),
        str(source_dir / "ppg_kernels.cu"),
        str(source_dir / "rcd_kernels.cu"),
        str(source_dir / "postprocess_kernels.cu"),
        str(source_dir / "laplacian.cu"),
        str(source_dir / "color_conversions.cu")
    ]
    
    print("Compiling CUDA extension...")
    return cpp_extension.load(
        name="torch_darktable_extension",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"] if not debug else ["-O0", "-g3", "-ggdb3"],
        verbose=debug,
        extra_cuda_cflags=["-G", "-O0",  '-lineinfo'] 
          if debug else ["-O3", "--expt-relaxed-constexpr", "--use_fast_math"]
    )

# Load extension on import
extension = _load_cuda_extension()

    
def ppg_demosaic(
              device: torch.device,
              image_size: tuple[int, int],
              bayer_pattern: BayerPattern,
              median_threshold: float | None = None) -> torch.Tensor:
    """
    Create a PPG demosaic algorithm object.    
    Args:
        raw_image: Input raw image tensor of shape (H, W) or (H, W, 1)
                  Must be on CUDA device and float32 dtype
        bayer_pattern: Bayer pattern, either string ('RGGB', 'BGGR', 'GRBG', 'GBRG') 
                      or integer filter code
        median_threshold: Threshold for pre-median filtering (0.0 to disable)
                        
    Returns:
        PPA demosaic algorithm object
    """
    return extension.PPG(device, image_size[0], image_size[1],
                        bayer_pattern.value,
                        median_threshold if median_threshold is not None else 0.0)


def rcd_demosaic(
              device: torch.device,
              image_size: tuple[int, int],
              bayer_pattern: BayerPattern,
              input_scale: float = 1.0,
              output_scale: float = 1.0) -> torch.Tensor:
    """
    Create a RCD demosaic algorithm object.
    
    Args:
        raw_image: Input raw image tensor of shape (H, W) or (H, W, 1)
                  Must be on CUDA device and float32 dtype
        bayer_pattern: Bayer pattern, either string ('RGGB', 'BGGR', 'GRBG', 'GBRG') 
                      or integer filter code
        input_scale: Scaling factor applied to input data (default 1.0)
        output_scale: Scaling factor applied to output data (default 1.0)
                        
    Returns:
        RCD demosaic algorithm object
    """
    
        
    return extension.RCD(device, image_size[0], image_size[1], bayer_pattern.value, input_scale, output_scale)


def postprocess_demosaic(
              device: torch.device,
              image_size: tuple[int, int],
              bayer_pattern: BayerPattern,
              color_smoothing_passes: int = 0,
              green_eq_local: bool = False,
              green_eq_global: bool = False,
              green_eq_threshold: float = 0.04) -> torch.Tensor:
    """
    Create a post-process demosaic algorithm object.

    Args:
        image_size: Image size (width, height)
        device: CUDA device
        bayer_pattern: Bayer pattern used for green equilibration
        color_smoothing_passes: Number of color smoothing passes (0 to disable)
        green_eq_local: Enable local green equilibration
        green_eq_global: Enable global green equilibration
        green_eq_threshold: Threshold for green equilibration (default 0.04, equivalent to 0.0001 * ISO 400)

    Returns:
        Post-process demosaic algorithm object
    """
    
    return extension.PostProcess(device, image_size[0], image_size[1],
                                bayer_pattern.value, color_smoothing_passes,
                                green_eq_local, green_eq_global, green_eq_threshold)



def create_laplacian(
    device: torch.device,
    image_size: tuple[int, int],
    params: LaplacianParams | None = None):
    """
    Create a reusable Laplacian filter for local tone mapping.

    Args:
        device: CUDA device for processing
        image_size: Image size (width, height) that will be processed
        params: LaplacianParams object (uses defaults if None)

    Returns:
        Laplacian workspace object that can be reused for multiple images
    """
    if params is None:
        params = LaplacianParams()

    print(f"Creating Laplacian filter for {image_size[0]}x{image_size[1]} with params {params}")

    assert image_size[0] > 0 and image_size[1] > 0, "Width and height must be positive"

    return extension.Laplacian(
        device, image_size[0], image_size[1], 
        params.num_gamma, 
        params.sigma, params.shadows, params.highlights, params.clarity)


def local_laplacian_rgb(
    laplacian: extension.Laplacian,
    image: torch.Tensor,
    ) -> torch.Tensor:
    """
    Apply local Laplacian filtering to RGB image.

    This function:
    1. Extracts luminance from RGB using LAB color space
    2. Processes luminance with local Laplacian filter
    3. Reconstructs RGB with modified luminance

    Args:
        image: Input RGB image tensor of shape (H, W, 3)
               Must be on CUDA device and float32 dtype, values 0-1
        laplacian: Laplacian workspace object created by create_laplacian()

    Returns:
        Filtered RGB image of same shape and type as input
    """
    assert image.dim() == 3 and image.size(2) == 3, "Input must be 3D tensor (H, W, 3)"
    assert image.device.type == 'cuda', "Input must be on CUDA device"
    assert image.dtype == torch.float32, "Input must be float32 dtype"

    # Extract luminance
    luminance = extension.compute_luminance(image)

    # Ensure luminance is contiguous and properly aligned
    luminance = luminance.contiguous()

    # Process luminance
    processed_luminance = laplacian.process(luminance)

    # Reconstruct RGB with modified luminance
    return extension.modify_luminance(image, processed_luminance)




__all__ = [
    "BayerPattern", "LaplacianParams",
    "create_laplacian", "local_laplacian_rgb"
]