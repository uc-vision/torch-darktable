"""JPEG encoding using NVJPEG."""

from enum import IntEnum

from .extension import extension

JpegException = extension.JpegException


class InputFormat(IntEnum):
  BGR = extension.JpegInputFormat.BGR
  RGB = extension.JpegInputFormat.RGB
  BGRI = extension.JpegInputFormat.BGRI
  RGBI = extension.JpegInputFormat.RGBI


class Subsampling(IntEnum):
  CSS_444 = extension.JpegSubsampling.CSS_444
  CSS_422 = extension.JpegSubsampling.CSS_422
  CSS_GRAY = extension.JpegSubsampling.CSS_GRAY


class Jpeg:
  def __init__(self):
    self.jpeg = extension.Jpeg()

  def encode(
    self, image, quality=94, input_format=InputFormat.RGBI, subsampling=Subsampling.CSS_422, progressive=False
  ):
    return self.jpeg.encode(image, quality, int(input_format), int(subsampling), progressive)


__all__ = ['InputFormat', 'Jpeg', 'JpegException', 'Subsampling']
