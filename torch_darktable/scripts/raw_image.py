
from dataclasses import dataclass
from pathlib import Path
import torch
from beartype import beartype
import torch_darktable as td
import cv2

@beartype
@dataclass
class CameraSettings:
    width: int
    ids_format: bool
    color_scales: tuple[float, float, float]
    brightness: float

@beartype
def load_raw_bytes(filepath:Path, device:torch.device=torch.device('cuda')):
  """Load raw image bytes into torch tensor without any decoding"""
  with open(filepath, 'rb') as f:
    raw_bytes = f.read()
  return torch.frombuffer(raw_bytes, dtype=torch.uint8).to(device, non_blocking=True)

def load_raw_image(filepath:Path, camera_settings:CameraSettings, device:torch.device=torch.device('cuda')) -> torch.Tensor:
  raw_cuda = load_raw_bytes(filepath, device).to(device, non_blocking=True)
  fmt = td.Packed12Format.IDS if camera_settings.ids_format else td.Packed12Format.STANDARD
  decoded = td.decode12(raw_cuda, output_dtype=torch.float32, format_type=fmt)

  bayer = decoded.view(-1, camera_settings.width)
  return scale_bayer(bayer, camera_settings.color_scales) * camera_settings.brightness



camera_settings = dict(
    blackfly=CameraSettings(width=4096, ids_format=False, color_scales=(1.0, 1.0, 1.0), brightness=0.8),
    ids=CameraSettings(width=2472, ids_format=True, color_scales=(1.5, 1.0, 1.5), brightness=1.0)
)


def add_camera_settings(parser):
    parser.add_argument('--camera', default='blackfly', choices=list(camera_settings.keys()), help='Camera to use')


def settings_from_args(args) -> CameraSettings:
  return camera_settings[args.camera]



def display_rgb(k, rgb_image):
  if isinstance(rgb_image, torch.Tensor):
    rgb_image = rgb_image.cpu().numpy()
  cv2.namedWindow(k, cv2.WINDOW_NORMAL)
  cv2.imshow(k, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
  cv2.waitKey(1)


def stack_bayer(bayer_image):
  return  torch.stack((
    bayer_image[0::2, 0::2], # Red
    bayer_image[0::2, 1::2], # Green
    bayer_image[1::2, 0::2], # Green
    bayer_image[1::2, 1::2] # Blue
  ), dim=-1)


def expand_bayer(x):
    h, w = x.shape[0], x.shape[1]
    result = torch.zeros(h * 2, w * 2, device=x.device, dtype=x.dtype)

    r, g1, g2, b = x.unbind(dim=-1)

    result[0::2, 0::2] = r  # Red
    result[0::2, 1::2] = g1  # Green
    result[1::2, 0::2] = g2  # Green
    result[1::2, 1::2] = b  # Blue
    return result


def scale_bayer(x, color_scales=(0.5, 1.0, 0.5)):
  r, g, b = color_scales
  scaling = torch.tensor([r, g, g, b], device=x.device, dtype=x.dtype)


  x = stack_bayer(x) * scaling
  x = expand_bayer(x)
  return x