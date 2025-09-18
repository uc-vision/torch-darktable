import torch
import argparse
from pathlib import Path
import cv2
import numpy as np

import torch_darktable as td


def load_rgb_image(image_path: Path) -> torch.Tensor:
  if not image_path.exists():
    raise FileNotFoundError(f'Image not found: {image_path}')
  img_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

  rgb_array = img_array.astype(np.float32)
  rgb_array = rgb_array / max(float(np.max(rgb_array)), 1.0)
  return torch.from_numpy(rgb_array).cuda()


def get_noise_level(args, input_rgb):
  if args.estimate_noise:
    noise_level = td.estimate_channel_noise(input_rgb)
    print(f'Estimated per-channel noise: R={noise_level[0]:.4f}, G={noise_level[1]:.4f}, B={noise_level[2]:.4f}')
    return noise_level * args.noise_scale
  else:
    print(f'Using manual noise level: {args.sigma}')
    return args.sigma


def test_denoise(image_path: Path, args):
  print(f'Loading image: {image_path}')
  input_rgb = load_rgb_image(image_path)
  height, width, channels = input_rgb.shape
  print(f'Image size: {width}x{height}x{channels}')

  # Keep in HWC format (no conversion needed)
  print('Creating Wiener denoiser...')
  wiener = td.create_wiener(input_rgb.device, (width, height), overlap=args.overlap)

  print('Processing...')
  with torch.no_grad():
    # Manual mode using provided sigma
    result_rgb = wiener.process(input_rgb, get_noise_level(args, input_rgb))

  # Already in HWC format

  input_display = (input_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
  result_display = (result_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
  combined = np.hstack([input_display, result_display])

  print('Showing results (original | denoised)...')
  cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
  cv2.imshow('Image', combined)
  while cv2.waitKey(1):
    pass
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(description='Test CUDA Wiener denoiser on an image')
  parser.add_argument('image', type=Path, help='Input image path')
  parser.add_argument(
    '--sigma',
    type=float,
    default=0.05,
    help='Noise standard deviation (manual mode)',
  )
  parser.add_argument(
    '--overlap',
    type=int,
    choices=[1, 2, 4, 8, 16],
    default=4,
    help='Overlap factor: 2=half, 4=quarter, 8=eighth block overlap',
  )

  parser.add_argument(
    '--estimate-noise',
    action='store_true',
    help='Use automatic per-channel noise estimation',
  )
  parser.add_argument(
    '--noise-scale',
    type=float,
    default=1.0,
    help='Scale factor for noise estimation (auto mode only)',
  )

  args = parser.parse_args()
  test_denoise(args.image, args)


if __name__ == '__main__':
  main()
