"""Tone mapping algorithms and utilities."""

import torch
from .extension import extension


class Reinhard:
    """Reinhard tone mapping algorithm with utilities for working with image metrics."""
    
    @staticmethod
    def compute_metrics(image: torch.Tensor, stride: int = 8) -> torch.Tensor:
        """
        Compute image statistics for Reinhard tone mapping.
        
        Args:
            image: Input RGB image tensor of shape (H, W, 3), float32, 0-1 range
            stride: Sampling stride for performance (default 8)
            
        Returns:
            9-element tensor containing image statistics
        """
        assert image.dim() == 3 and image.size(2) == 3, "Input must be (H, W, 3)"
        assert image.dtype == torch.float32, "Input must be float32"
        assert image.device.type == 'cuda', "Input must be on CUDA device"
        
        return extension.compute_image_metrics(image, stride)
    
    @staticmethod
    def metrics_to_dict(metrics: torch.Tensor) -> dict[str, float]:
        """Convert 9-element metrics tensor to named dictionary."""
        assert metrics.numel() == 9, f"Expected 9 elements, got {metrics.numel()}"
        m = metrics.cpu().numpy()
        return {
            'bounds_min': float(m[0]),
            'bounds_max': float(m[1]),
            'log_bounds_min': float(m[2]),
            'log_bounds_max': float(m[3]),
            'log_mean': float(m[4]),
            'mean': float(m[5]),
            'rgb_mean_r': float(m[6]),
            'rgb_mean_g': float(m[7]),
            'rgb_mean_b': float(m[8])
        }
    
    @staticmethod
    def metrics_from_dict(metrics_dict: dict[str, float], device: torch.device = torch.device('cuda')) -> torch.Tensor:
        """Convert named dictionary to 9-element metrics tensor."""
        return torch.tensor([
            metrics_dict['bounds_min'], metrics_dict['bounds_max'],
            metrics_dict['log_bounds_min'], metrics_dict['log_bounds_max'],
            metrics_dict['log_mean'], metrics_dict['mean'],
            metrics_dict['rgb_mean_r'], metrics_dict['rgb_mean_g'], metrics_dict['rgb_mean_b']
        ], device=device, dtype=torch.float32)
    
    @staticmethod
    def print_metrics(metrics: torch.Tensor):
        """Print metrics in a nicely formatted way."""
        d = Reinhard.metrics_to_dict(metrics)
        print("Reinhard Image Metrics:")
        print(f"  Bounds: [{d['bounds_min']:.4f}, {d['bounds_max']:.4f}]")
        print(f"  Log Bounds: [{d['log_bounds_min']:.4f}, {d['log_bounds_max']:.4f}]")
        print(f"  Log Mean: {d['log_mean']:.4f}")
        print(f"  Mean: {d['mean']:.4f}")
        print(f"  RGB Mean: ({d['rgb_mean_r']:.4f}, {d['rgb_mean_g']:.4f}, {d['rgb_mean_b']:.4f})")
    
    @staticmethod 
    def tonemap(
        image: torch.Tensor,
        metrics: torch.Tensor,
        gamma: float = 1.0,
        intensity: float = 1.0,
        light_adapt: float = 0.8
    ) -> torch.Tensor:
        """
        Apply Reinhard tone mapping to HDR image.
        
        Args:
            image: Input RGB image tensor of shape (H, W, 3), float32
            metrics: 9-element tensor of image statistics
            gamma: Gamma correction factor (default 1.0)
            intensity: Overall exposure control (default 1.0)
            light_adapt: Local vs global adaptation blend (0=global, 1=local, default 0.8)
            
        Returns:
            Tone mapped image as uint8 tensor (H, W, 3) in range [0, 255]
        """
        assert image.dim() == 3 and image.size(2) == 3, "Input must be (H, W, 3)"
        assert image.dtype == torch.float32, "Input must be float32"
        assert image.device.type == 'cuda', "Input must be on CUDA device"
        assert metrics.numel() == 9, "Metrics tensor must have 9 elements"
        
        return extension.reinhard_tonemap(image, metrics, gamma, intensity, light_adapt)


def aces_tonemap(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """
    Apply ACES tone mapping (industry standard).
    
    Note: This doesn't use any image statistics, so exposure may need manual adjustment.
    
    Args:
        image: Input RGB image tensor of shape (H, W, 3), float32
        gamma: Gamma correction factor (default 2.2)
        
    Returns:
        Tone mapped image as uint8 tensor (H, W, 3) in range [0, 255]
    """
    assert image.dim() == 3 and image.size(2) == 3, "Input must be (H, W, 3)"
    assert image.dtype == torch.float32, "Input must be float32"
    assert image.device.type == 'cuda', "Input must be on CUDA device"
    
    return extension.aces_tonemap(image, gamma)


compute_image_bounds = extension.compute_image_bounds


__all__ = ["Reinhard", "aces_tonemap", "compute_image_bounds"]
