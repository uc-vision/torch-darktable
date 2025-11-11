from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = Path(__file__).parent.resolve()
abs_include = str(here / 'torch_darktable' / 'csrc')

ext = CUDAExtension(
  name='torch_darktable.torch_darktable_extension',
  sources=[
    f'torch_darktable/csrc/{s}'
    for s in [
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
  ],
  include_dirs=[abs_include],
  libraries=['nvjpeg'],
  extra_compile_args={
    'cxx': ['-O3', '-std=c++17', '-Wno-stringop-overread'],
    'nvcc': ['-O3', '--expt-relaxed-constexpr', '--use_fast_math', '-std=c++17'],
  },
)

setup(
  ext_modules=[ext],
  cmdclass={'build_ext': BuildExtension},
  package_data={'torch_darktable': ['*.pyi', 'camera_settings/*.json']},
)
