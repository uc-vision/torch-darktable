import argparse
from pathlib import Path

import cv2
import torch

from torch_darktable import InputFormat, Jpeg
from torch_darktable.scripts.util import display_rgb


def test_jpeg_encode(image_path: Path, quality: int, progressive: bool, input_format: InputFormat):
  print(f'Loading image: {image_path}')

  original_rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
  assert original_rgb is not None, f'Could not load image {image_path}'

  original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
  height, width = original_rgb.shape[:2]

  print(f'Image size: {width}x{height}')
  print(f'Quality: {quality}')
  print(f'Progressive: {progressive}')
  print(f'Input format: {input_format.name}')

  if input_format in {InputFormat.BGRI, InputFormat.RGBI}:
    image_np = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR) if input_format == InputFormat.BGRI else original_rgb
    image_u8 = torch.from_numpy(image_np).cuda()
  else:
    image_np = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR) if input_format == InputFormat.BGR else original_rgb
    image_u8 = torch.from_numpy(image_np).permute(2, 0, 1).cuda()

  jpeg = Jpeg()
  print('Encoding JPEG...')
  encoded = jpeg.encode(image_u8, quality=quality, input_format=input_format, progressive=progressive)

  encoded_size = encoded.numel()
  print(f'Encoded size: {encoded_size} bytes ({encoded_size / 1024:.2f} KB)')

  encoded_cpu = encoded.cpu().numpy()
  print('Decoding JPEG with OpenCV...')
  decoded = cv2.imdecode(encoded_cpu, cv2.IMREAD_COLOR)
  assert decoded is not None

  decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
  display_rgb('JPEG Decoded', decoded_rgb)


def main():
  parser = argparse.ArgumentParser(description='Test JPEG encoding')
  parser.add_argument('image', type=Path, help='Input image path')
  parser.add_argument('--quality', type=int, default=94, help='JPEG quality (1-100, default: 94)')
  parser.add_argument('--progressive', action='store_true', help='Use progressive JPEG')
  parser.add_argument(
    '--format',
    type=str,
    default='BGRI',
    choices=['BGR', 'RGB', 'BGRI', 'RGBI'],
    help='Input format (default: BGRI)',
  )

  args = parser.parse_args()

  input_format = InputFormat[args.format]
  test_jpeg_encode(args.image, args.quality, args.progressive, input_format)


if __name__ == '__main__':
  main()
