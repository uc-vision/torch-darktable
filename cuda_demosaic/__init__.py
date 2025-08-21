"""
PPG Demosaic - GPU-accelerated demosaicing for PyTorch

This package provides a standalone implementation of the PPG (Pattern Pixel Grouping) 
demosaic algorithm extracted from darktable, optimized for GPU execution with PyTorch.
"""

from .demosaic import ppg_demosaic, rcd_demosaic, postprocess_demosaic, BayerPattern
__all__ = ["ppg_demosaic", "rcd_demosaic", "postprocess_demosaic", "BayerPattern"]
