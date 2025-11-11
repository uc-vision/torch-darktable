"""Dump all camera settings to JSON files."""

from pathlib import Path

from torch_darktable.pipeline.camera_settings import load_camera_settings_from_dir


def main():
  import argparse

  parser = argparse.ArgumentParser(description='Dump camera settings to JSON files')
  parser.add_argument('output_dir', type=Path, help='Output directory for JSON files')
  args = parser.parse_args()

  args.output_dir.mkdir(parents=True, exist_ok=True)

  camera_settings = load_camera_settings_from_dir()
  for name, settings in camera_settings.items():
    output_file = args.output_dir / f'{name}.json'
    settings.save_json(output_file)
    print(f'Wrote {name} to {output_file}')
