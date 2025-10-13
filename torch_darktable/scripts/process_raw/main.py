import argparse
from pathlib import Path

import torch

from torch_darktable.scripts.util import CameraSettings, camera_settings, settings_for_file

from .pipeline_ui import PipelineController
from .ui import ProcessRawUI


def find_image_files(image_path: Path) -> list[Path]:
  """Find all image files with the same extension in the same directory."""
  directory = image_path.parent
  extension = image_path.suffix
  return sorted(directory.glob(f'*{extension}'))


def parse_args():
  parser = argparse.ArgumentParser(description='Run inference on raw images using ZRR models')
  parser.add_argument('input', type=Path, help='Path to input raw image')

  parser.add_argument(
    '--camera', type=str, default=None, help='Camera name (one of ' + ', '.join(camera_settings.keys()) + ')'
  )
  parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for JPEG files (default: /tmp)')
  parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda:0)')

  return parser.parse_args()


def interactive_debayer(
  image_files: list[Path],
  current_index: int,
  camera_settings: CameraSettings,
  device: torch.device,
  output_dir: Path | None = None,
) -> None:
  """Interactive raw image processing with navigation."""
  camera_name = image_files[current_index].parent.stem
  image_transform = camera_settings.get_image_transform(camera_name)
  pipeline_controller = PipelineController(camera_settings=camera_settings, device=device, image_transform=image_transform)
  ui = ProcessRawUI(image_files, current_index, pipeline_controller, output_dir)
  ui.show()


def main():
  args = parse_args()
  assert args.input.exists() and args.input.is_file(), f'Error: Input file {args.input} does not exist'

  # Get camera settings
  if args.camera:
    if args.camera not in camera_settings:
      raise ValueError(f'Unknown camera: {args.camera}. Available cameras: {list(camera_settings.keys())}')
    cam_settings = camera_settings[args.camera]
  else:
    cam_settings = settings_for_file(args.input)

  # Find all images with same extension
  image_files = find_image_files(args.input)
  current_index = image_files.index(args.input)

  if len(image_files) > 1:
    print(f'Found {len(image_files)} images with extension {args.input.suffix}')

  # Print output directory info
  output_dir = args.output_dir if args.output_dir is not None else Path('/tmp')
  print(f'JPEG files will be saved to: {output_dir.absolute()}')

  interactive_debayer(image_files, current_index, cam_settings, torch.device(args.device), args.output_dir)


if __name__ == '__main__':
  main()
