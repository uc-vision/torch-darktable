from pathlib import Path

import cv2
import numpy as np
import torch


def load_image(image_path: Path) -> torch.Tensor:
  img_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
  assert img_array is not None, f'Could not load image {image_path}'
  rgb_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  return torch.from_numpy(rgb_array).cuda()


def display_rgb(k: str, rgb_image: torch.Tensor | np.ndarray):
  if isinstance(rgb_image, torch.Tensor):
    rgb_image = rgb_image.cpu().numpy()
  cv2.namedWindow(k, cv2.WINDOW_NORMAL)

  # loop while wiow is not closed
  cv2.imshow(k, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
  while cv2.waitKey(1) & 255 != ord('q') or cv2.getWindowProperty(k, cv2.WND_PROP_VISIBLE) >= 1:
    pass

  cv2.destroyAllWindows()
