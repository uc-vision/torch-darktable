"""
PPG Demosaic - GPU-accelerated demosaicing for PyTorch

This package provides a standalone implementation of the PPG (Pattern Pixel Grouping) 
demosaic algorithm extracted from darktable, optimized for GPU execution with PyTorch.
"""

from .ppg_demosaic import PPGDemosaic, ppg_demosaic

__version__ = "0.1.0"
__author__ = "PPG Extractor"
__all__ = ["PPGDemosaic", "ppg_demosaic"]
