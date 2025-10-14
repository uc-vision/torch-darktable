"""JPEG encoding/decoding utilities for quality assessment."""

from pathlib import Path

import cv2
import numpy as np


def _get_jpeg_encode_params(quality: int, progressive: bool) -> list:
  """Get OpenCV JPEG encoding parameters."""
  params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
  if progressive:
    params.extend([cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
  return params


def encode_decode_jpeg(rgb_array, quality: int = 95, progressive: bool = False):
  """Encode and decode JPEG in memory, return (decoded_rgb, size_bytes)."""
  bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
  params = _get_jpeg_encode_params(quality, progressive)

  success, encoded_img = cv2.imencode('.jpg', bgr_array, params)
  if not success:
    raise RuntimeError('Failed to encode JPEG')

  decoded_bgr = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
  decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

  return decoded_rgb, len(encoded_img.tobytes())


def save_jpeg_to_disk(rgb_array: np.ndarray, output_path: Path, quality: int = 95, progressive: bool = False):
  """Save JPEG to disk."""
  bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
  params = _get_jpeg_encode_params(quality, progressive)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(output_path), bgr_array, params)


def calculate_psnr(original, compressed) -> float:
  """Calculate PSNR between original and compressed images."""
  mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
  if mse == 0:
    return float('inf')
  return 20 * np.log10(255.0 / np.sqrt(mse))
