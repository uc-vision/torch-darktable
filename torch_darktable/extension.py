"""CUDA extension loading and basic types."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils import cpp_extension


def _load_cuda_extension(debug: bool = False, verbose: bool = False):
  source_dir = Path(__file__).parent / 'csrc'
  source_files = [
    'extension.cpp',
    'debayer/bilinear.cu',
    'debayer/ppg.cu',
    'debayer/rcd.cu',
    'debayer/postprocess.cu',
    'local_contrast/laplacian.cu',
    'local_contrast/bilateral.cu',
    'color_conversions.cu',
    'packed.cu',
    'tonemap/color_adaption.cu',
    'tonemap/aces.cu',
    'tonemap/linear.cu',
    'tonemap/reinhard.cu',
    'white_balance.cu',
    'denoise/denoise.cu',
    'jpeg_encoder.cu',
  ]
  sources = [str(source_dir / f) for f in source_files]

  extension_name = 'torch_darktable_extension_debug' if debug else 'torch_darktable_extension'
  return cpp_extension.load(
    name=extension_name,
    sources=sources,
    extra_include_paths=[str(source_dir)],
    extra_cflags=['-O3', '-std=c++17', '-Wno-stringop-overread']
    if not debug
    else ['-O0', '-g3', '-ggdb3', '-Wno-stringop-overread'],
    verbose=verbose,
    with_cuda=True,
    extra_cuda_cflags=['-G', '-O0', '-lineinfo']
    if debug
    else ['-O3', '--expt-relaxed-constexpr', '--use_fast_math', '-std=c++17'],
    extra_ldflags=['-lnvjpeg'],
  )


# Load extension on import, with static types for type checkers
if TYPE_CHECKING:
  from . import torch_darktable_extension as extension  # type: ignore
else:
  debug_mode = os.getenv('TORCH_DARKTABLE_DEBUG', '0') == '1'
  print(f'Debug mode: {"enabled" if debug_mode else "disabled"}')
  extension = _load_cuda_extension(debug=debug_mode, verbose=True)


# readable representations for algorithm classes
def _install_algorithm_repr() -> None:
  def _ppg_repr(self) -> str:
    return f'PPG({self.width}x{self.height}, median_threshold={self.median_threshold})'

  def _rcd_repr(self) -> str:
    return f'RCD({self.width}x{self.height})'

  def _postprocess_repr(self) -> str:
    return (
      f'PostProcess({self.width}x{self.height}, color_smoothing_passes={self.color_smoothing_passes}, '
      f'green_eq_local={self.green_eq_local}, '
      f'green_eq_global={self.green_eq_global}, '
      f'green_eq_threshold={self.green_eq_threshold})'
    )

  def _laplacian_repr(self) -> str:
    return f'Laplacian({self.width}x{self.height}, sigma={self.sigma}, shadows={self.shadows}, highlights={self.highlights}, clarity={self.clarity})'

  def _bilateral_repr(self) -> str:
    return f'Bilateral({self.width}x{self.height}, sigma_s={self.sigma_s}, sigma_r={self.sigma_r})'

  extension.PPG.__repr__ = _ppg_repr
  extension.RCD.__repr__ = _rcd_repr
  extension.PostProcess.__repr__ = _postprocess_repr
  extension.Laplacian.__repr__ = _laplacian_repr
  extension.Bilateral.__repr__ = _bilateral_repr


_install_algorithm_repr()

__all__ = ['extension']
