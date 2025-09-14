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

def check_overlap_factor(overlap_factor: int):
    if overlap_factor not in [1, 2, 4, 8, 16]:
        raise ValueError("overlap_factor must be 1, 2, 4, 8, or 16")


class Wiener:
    """High-level Wiener denoiser with flexible noise parameter handling."""
    

    def __init__(self, device: torch.device, image_size: tuple[int, int], 
                 eps: float = 1e-15, overlap_factor: int = 4):
        width, height = image_size
        check_overlap_factor(overlap_factor)

        self._wiener = extension.Wiener(device, width, height, eps, overlap_factor)
        self._device = device

    @property
    def eps(self) -> float:
        return self._wiener.eps

    @eps.setter
    def eps(self, eps: float):
        self._wiener.eps = eps

    @property
    def overlap_factor(self) -> int:
        return self._wiener.overlap_factor

    @overlap_factor.setter
    def overlap_factor(self, overlap_factor: int):
        check_overlap_factor(overlap_factor)
        self._wiener.overlap_factor = overlap_factor
    
    @beartype
    def process(self, image: torch.Tensor, noise: float | torch.Tensor | None = None) -> torch.Tensor:
        """
        Process image with Wiener filter.
        
        Args:
            image: RGB image tensor of shape (H, W, 3)
            noise: Noise levels - float (all channels), Tensor[1] (all channels), 
                   Tensor[3] (per-channel), or None (auto-estimate)
                   
        Returns:
            Denoised image of same shape
        """
        if noise is None:
            # Auto-estimate per-channel noise
            noise_sigmas = estimate_channel_noise(image)
        elif isinstance(noise, float):
            # Single float for all channels
            noise_sigmas = torch.tensor([noise, noise, noise], 
                                       dtype=torch.float32, device=self._device)
        elif isinstance(noise, torch.Tensor):
            if noise.shape != (3,):
                raise ValueError("noise tensor must have 3 elements")
            noise_sigmas = noise.to(dtype=torch.float32, device=self._device)
           
        return self._wiener.process(image, noise_sigmas)


@beartype
def create_wiener(
    device: torch.device,
    image_size: tuple[int, int],
    eps: float = 1e-15,
    overlap_factor: int = 4
) -> Wiener:
    """
    Create a Wiener denoiser object with flexible noise handling.
    
    Args:
        device: CUDA device to use
        image_size: (width, height) of the image
        eps: Regularization epsilon
        overlap_factor: Overlap factor (2=half, 4=quarter, 8=eighth block overlap)
        
    Returns:
        High-level Wiener denoiser object
    """
    return Wiener(device, image_size, eps, overlap_factor)


@beartype
def estimate_channel_noise(image: torch.Tensor) -> torch.Tensor:
    """
    Estimate per-channel noise levels from image using high-frequency analysis.
    
    Based on paper's MAD approach: σ ≈ median(|H - median(H)|) / 0.6745
    
    Args:
        image: RGB image tensor of shape (H, W, 3)
        
    Returns:
        Per-channel noise sigmas as tensor of shape (3,) for R, G, B channels
    """
    # Laplacian kernel for high-frequency extraction
    laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                            dtype=image.dtype, device=image.device)
    
    # Apply to all channels at once: (H, W, 3) → (1, 3, H, W) for conv2d
    image_chw = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    laplacian_3ch = laplacian.unsqueeze(0).repeat(3, 1, 1).unsqueeze(1)  # (3, 1, 3, 3)
    
    # 3-channel convolution (one kernel per channel)
    high_freq = torch.conv2d(image_chw, laplacian_3ch, groups=3, padding=1)  # (1, 3, H, W)
    
    # Compute MAD per channel
    noise_sigmas = torch.zeros(3, dtype=image.dtype, device=image.device)
    for c in range(3):
        channel_high_freq = high_freq[0, c].flatten()
        median_val = torch.median(channel_high_freq)
        mad = torch.median(torch.abs(channel_high_freq - median_val))
        noise_sigmas[c] = mad / 0.6745
    
    return noise_sigmas


