import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import torch_darktable as td
from torch_darktable import LaplacianParams, compute_luminance


def reinhard(image: torch.Tensor, epsilon=1e-4, base_key=0.18, gamma=0.75) -> torch.Tensor:
  lum = compute_luminance(image)
  log_avg = torch.exp(torch.log(lum + epsilon).mean())
  key = base_key / log_avg

  scaled = image * key

  tonemapped = (scaled / (1.0 + scaled)) ** gamma
  return tonemapped.clamp(0.0, 1.0)


def load_rgb_image(image_path: Path) -> torch.Tensor:
  """Load RGB image as tensor."""
  if not image_path.exists():
    raise FileNotFoundError(f'Image not found: {image_path}')

  img_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # -1 preserves bit depth

  rgb_array = img_array.astype(np.float32)
  print(f'range: {np.min(rgb_array):.3f} - {np.max(rgb_array):.3f}')

  rgb_array /= np.max(rgb_array)

  return torch.from_numpy(rgb_array).cuda()


def test_laplacian(image_path: Path, args):
  """Test local Laplacian filter on RGB image using LAB color space processing."""
  print(f'Loading image: {image_path}')

  # Load RGB image
  input_rgb = load_rgb_image(image_path)
  height, width, channels = input_rgb.shape

  print(f'Image size: {width}x{height}x{channels}')
  print(f'RGB range: {input_rgb.min().item():.3f} - {input_rgb.max().item():.3f}')

  # Create Laplacian algorithm from args
  print('Creating Laplacian algorithm...')
  params = LaplacianParams(
    num_gamma=args.num_gamma,
    sigma=args.sigma,
    shadows=args.shadows,
    highlights=args.highlights,
    clarity=args.clarity,
  )
  workspace = td.Laplacian(input_rgb.device, (width, height), params)

  print(
    f'Parameters: gamma={params.num_gamma}, sigma={params.sigma},'
    f'shadows={params.shadows}, highlights={params.highlights}, clarity={params.clarity}'
  )

  print('Applying local Laplacian filter to RGB image...')
  result_rgb = workspace.process_rgb(input_rgb)

  if args.tonemap:
    input_rgb = reinhard(input_rgb)
    result_rgb = reinhard(result_rgb)

  # Convert results to displayable format
  input_display = (input_rgb.cpu().numpy() * 255).astype(np.uint8)
  result_display = (result_rgb.cpu().numpy() * 255).astype(np.uint8)

  # Create side-by-side comparison
  combined = np.hstack([input_display, result_display])

  print('Showing results (original | filtered)...')
  print('Press any key to close')

  cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
  cv2.imshow('Image', combined)
  while cv2.waitKey(1):
    pass
  cv2.destroyAllWindows()

  print(f'Output RGB range: {result_rgb.min().item():.3f} - {result_rgb.max().item():.3f}')


def main():
  parser = argparse.ArgumentParser(description='Test local Laplacian filter on an image')
  parser.add_argument('image', type=Path, help='Input image path')
  parser.add_argument('--num_gamma', type=int, default=6, help='Number of gamma levels (4, 6, or 8)')
  parser.add_argument(
    '--sigma',
    type=float,
    default=0.2,
    help='Tone mapping parameter controlling transitions (default: 0.2)',
  )
  parser.add_argument('--shadows', type=float, default=1.0, help='Shadow enhancement, default: 1.0')
  parser.add_argument(
    '--highlights',
    type=float,
    default=1.0,
    help='Highlight compression, default: 1.0',
  )
  parser.add_argument(
    '--clarity',
    type=float,
    default=0.0,
    help='Local contrast enhancement, default: 0.0',
  )

  parser.add_argument('--tonemap', action='store_true', help='Tonemap the output image')

  args = parser.parse_args()

  test_laplacian(args.image, args)


if __name__ == '__main__':
  main()
