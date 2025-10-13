import argparse
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from torch_darktable.scripts.util import cv2, display_rgb

models = dict(
  small='depth-anything/Depth-Anything-V2-Small-hf',
)


@dataclass
class DepthEstimation:
  model: torch.nn.Module
  image_processor: AutoImageProcessor

  @cached_property
  def compiled_model(self):
    return torch.compile(self.model.forward)

  @staticmethod
  def create_model(name: str = 'small'):
    assert name in models, f'Model {name} not found. Available models: {models.keys()}'

    image_processor = AutoImageProcessor.from_pretrained(models[name], use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(models[name])

    return DepthEstimation(model=model, image_processor=image_processor)

  def estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
    inputs = self.image_processor(images=image, return_tensors='pt')  # type: ignore

    outputs = self.compiled_model(**inputs)
    predicted_depth = outputs.predicted_depth.squeeze(0)

    return (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())


def get_color_map(name=cv2.COLORMAP_TURBO, device=torch.device('cuda:0')):
  table = np.arange(256, dtype=np.uint8)[:, None]
  colors = cv2.applyColorMap(table, name)

  return torch.from_numpy(colors).to(device).squeeze(1)


def color_map_image(image_gray: torch.Tensor, color_map_name=cv2.COLORMAP_TURBO):
  color_map = get_color_map(color_map_name)

  print(image_gray.shape, color_map.shape)

  if image_gray.dtype == torch.float32:
    image_gray = (image_gray * 255).to(torch.int32)

  return color_map[image_gray]


def main():
  args = argparse.ArgumentParser()
  args.add_argument('image', type=Path, help='Input image path')
  args.add_argument('--model', type=str, default='small', help='Model name')
  args = args.parse_args()

  depth_estimator = DepthEstimation.create_model(args.model)

  image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = torch.from_numpy(image).permute(2, 0, 1)

  depth = depth_estimator.estimate_depth(image)

  color_map = color_map_image(depth)
  display_rgb('Depth', color_map)


if __name__ == '__main__':
  main()
