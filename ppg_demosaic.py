"""
PPG Demosaic Python Interface
GPU-only implementation using PyTorch and OpenCL
"""

import torch
import numpy as np
from typing import Union, Optional
import ppg_demosaic_cuda  # The compiled C++ extension

class PPGDemosaic:
    """
    PPG (Pattern Pixel Grouping) Demosaic algorithm.
    
    This is a GPU-only implementation that processes Bayer pattern raw images
    and converts them to full-color images using the PPG algorithm from darktable.
    """
    
    # Bayer pattern filters (same as darktable)
    BAYER_RGGB = 0x94949494  # Red-Green-Green-Blue
    BAYER_BGGR = 0x16161616  # Blue-Green-Green-Red  
    BAYER_GRBG = 0x61616161  # Green-Red-Blue-Green
    BAYER_GBRG = 0x49494949  # Green-Blue-Red-Green
    
    def __init__(self):
        """Initialize PPG Demosaic processor"""
        pass
    
    @staticmethod
    def get_bayer_pattern(pattern: str) -> int:
        """
        Convert string bayer pattern to filter code.
        
        Args:
            pattern: One of 'RGGB', 'BGGR', 'GRBG', 'GBRG'
            
        Returns:
            Filter code for the pattern
        """
        patterns = {
            'RGGB': PPGDemosaic.BAYER_RGGB,
            'BGGR': PPGDemosaic.BAYER_BGGR, 
            'GRBG': PPGDemosaic.BAYER_GRBG,
            'GBRG': PPGDemosaic.BAYER_GBRG
        }
        
        if pattern not in patterns:
            raise ValueError(f"Unknown bayer pattern: {pattern}. Must be one of {list(patterns.keys())}")
        
        return patterns[pattern]
    
    def demosaic(self, 
                 raw_image: Union[torch.Tensor, np.ndarray],
                 bayer_pattern: Union[str, int],
                 median_threshold: float = 0.0) -> torch.Tensor:
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
        
        # Convert numpy to torch if needed
        if isinstance(raw_image, np.ndarray):
            raw_image = torch.from_numpy(raw_image).float()
            
        # Ensure tensor is on CUDA
        if not raw_image.device.type == 'cuda':
            raise ValueError("Input tensor must be on CUDA device")
            
        # Ensure correct dtype
        if raw_image.dtype != torch.float32:
            raw_image = raw_image.float()
            
        # Ensure correct shape (H, W, 1)
        if raw_image.dim() == 2:
            raw_image = raw_image.unsqueeze(-1)  # Add channel dimension
        elif raw_image.dim() == 3 and raw_image.size(2) != 1:
            raise ValueError("Input must have shape (H, W) or (H, W, 1)")
        elif raw_image.dim() != 3:
            raise ValueError("Input must be 2D or 3D tensor")
            
        # Convert bayer pattern to filter code
        if isinstance(bayer_pattern, str):
            filters = self.get_bayer_pattern(bayer_pattern)
        else:
            filters = int(bayer_pattern)
            
        # Call CUDA extension
        result = ppg_demosaic_cuda.ppg_demosaic_cuda(raw_image, filters, median_threshold)
        
        # Return only RGB channels (drop alpha)
        return result[:, :, :3]
    
    def __call__(self, raw_image: Union[torch.Tensor, np.ndarray], 
                 bayer_pattern: Union[str, int],
                 median_threshold: float = 0.0) -> torch.Tensor:
        """Convenience method for calling demosaic"""
        return self.demosaic(raw_image, bayer_pattern, median_threshold)


def ppg_demosaic(raw_image: Union[torch.Tensor, np.ndarray],
                 bayer_pattern: Union[str, int] = 'RGGB',
                 median_threshold: float = 0.0) -> torch.Tensor:
    """
    Convenience function for PPG demosaicing.
    
    Args:
        raw_image: Input raw image tensor (H, W) or (H, W, 1), must be on CUDA
        bayer_pattern: Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG') or filter code
        median_threshold: Pre-median filter threshold (0.0 to disable)
        
    Returns:
        Demosaiced RGB image (H, W, 3)
    """
    demosaicer = PPGDemosaic()
    return demosaicer.demosaic(raw_image, bayer_pattern, median_threshold)


# Example usage
if __name__ == "__main__":
    # Create dummy raw image on GPU
    device = torch.device('cuda', 0)

    # Create test raw image (512x512)
    raw = torch.rand(512, 512, device=device, dtype=torch.float32)
    
    # Demosaic using PPG
    demosaicer = PPGDemosaic()
    rgb = demosaicer.demosaic(raw, 'RGGB')
    
    print(f"Input shape: {raw.shape}")
    print(f"Output shape: {rgb.shape}")
    print(f"Output device: {rgb.device}")
    print(f"Output dtype: {rgb.dtype}")
