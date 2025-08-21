"""
PPG Demosaic Python Interface
GPU-only implementation using PyTorch and CUDA
"""

import torch
from enum import Enum
from pathlib import Path
from torch.utils import cpp_extension


class BayerPattern(Enum):
  RGGB = 0x94949494
  BGGR = 0x16161616
  GRBG = 0x61616161
  GBRG = 0x49494949


# Dynamic compilation of CUDA extension
def _load_cuda_extension():
    """Load the CUDA extension dynamically"""
    current_dir = Path(__file__).parent
    source_dir = current_dir / "csrc"
    
    sources = [
        str(source_dir / "demosaic.cpp"),
        str(source_dir / "ppg_kernels.cu"),
        str(source_dir / "rcd_kernels.cu")
    ]
    
    return cpp_extension.load(
        name="ppg_demosaic_cuda",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-arch=sm_70"],
        verbose=False
    )

# Load extension on import
demosaic_cuda = _load_cuda_extension()

    
def ppg_demosaic(
              raw_image: torch.Tensor,
              bayer_pattern: BayerPattern,
              median_threshold: float | None = None) -> torch.Tensor:
    """
    Demosaic a raw Bayer pattern image using PPG algorithm.
    
    Args:
        raw_image: Input raw image tensor of shape (H, W) or (H, W, 1)
                  Must be on CUDA device and float32 dtype
        bayer_pattern: Bayer pattern, either string ('RGGB', 'BGGR', 'GRBG', 'GBRG') 
                      or integer filter code
        median_threshold: Threshold for pre-median filtering (0.0 to disable)
                        
    Returns:
        Demosaiced RGB image of shape (H, W, 3)
    """
    
        
    assert raw_image.dim() == 3, "Input must be 3D tensor"
    assert raw_image.size(2) == 1, "Input must have shape (H, W, 1)"
    assert raw_image.device.type == 'cuda', "Input must be on CUDA device"
    assert raw_image.dtype == torch.float32, "Input must be float32 dtype"
    

    result = demosaic_cuda.ppg_demosaic(raw_image, bayer_pattern.value, 
      median_threshold if median_threshold is not None else 0.0)
    return result[:, :, :3]


def rcd_demosaic(
              raw_image: torch.Tensor,
              bayer_pattern: BayerPattern,
              input_scale: float = 1.0,
              output_scale: float = 1.0) -> torch.Tensor:
    """
    Demosaic a raw Bayer pattern image using RCD (Ratio Corrected Demosaicing) algorithm.
    
    Args:
        raw_image: Input raw image tensor of shape (H, W) or (H, W, 1)
                  Must be on CUDA device and float32 dtype
        bayer_pattern: Bayer pattern, either string ('RGGB', 'BGGR', 'GRBG', 'GBRG') 
                      or integer filter code
        input_scale: Scaling factor applied to input data (default 1.0)
        output_scale: Scaling factor applied to output data (default 1.0)
                        
    Returns:
        Demosaiced RGB image of shape (H, W, 3)
    """
    
        
    assert raw_image.dim() == 3, "Input must be 3D tensor"
    assert raw_image.size(2) == 1, "Input must have shape (H, W, 1)"
    assert raw_image.device.type == 'cuda', "Input must be on CUDA device"
    assert raw_image.dtype == torch.float32, "Input must be float32 dtype"
    

    result = demosaic_cuda.rcd_demosaic(raw_image, bayer_pattern.value, input_scale, output_scale)
    return result[:, :, :3]

