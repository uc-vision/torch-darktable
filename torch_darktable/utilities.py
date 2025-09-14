import torch
import numpy as np
from pathlib import Path
from PIL import Image
from .extension import extension
from beartype import beartype


def load_image(image_path: Path) -> torch.Tensor:
    """Load RGB image as torch tensor on GPU.
    
    Returns:
        RGB tensor of shape (H, W, 3) with values in [0, 1], on CUDA
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    rgb_array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(rgb_array).cuda()


def rgb_to_bayer(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to Bayer pattern (RGGB).
    
    Args:
        rgb_tensor: RGB tensor of shape (H, W, 3)
        
    Returns:
        Bayer tensor of shape (H, W, 1)
    """
    height, width = rgb_tensor.shape[:2]
    bayer = torch.zeros((height, width), dtype=rgb_tensor.dtype, device=rgb_tensor.device)
    
    # Create RGGB Bayer pattern
    bayer[0::2, 0::2] = rgb_tensor[0::2, 0::2, 0]  # R (even rows, even cols)
    bayer[0::2, 1::2] = rgb_tensor[0::2, 1::2, 1]  # G (even rows, odd cols)
    bayer[1::2, 0::2] = rgb_tensor[1::2, 0::2, 1]  # G (odd rows, even cols)  
    bayer[1::2, 1::2] = rgb_tensor[1::2, 1::2, 2]  # B (odd rows, odd cols)
    
    return bayer.unsqueeze(-1)


@beartype
def create_wiener(
    device: torch.device,
    image_size: tuple[int, int],
    sigma: float = 0.05,
    eps: float = 1e-15
) -> "extension.Wiener":
    """
    Create a Wiener denoiser object.
    
    Args:
        device: CUDA device to use
        image_size: (width, height) of the image
        sigma: Noise standard deviation
        eps: Regularization epsilon
        
    Returns:
        Wiener algorithm object
    """
    width, height = image_size
    return extension.Wiener(device, width, height, sigma, eps)


