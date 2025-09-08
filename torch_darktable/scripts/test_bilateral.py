import torch
import argparse
from pathlib import Path
import cv2
import numpy as np

from torch_darktable import create_bilateral, bilateral_rgb


def reinhard(image: torch.Tensor, epsilon=1e-4, base_key=0.18, gamma=0.75) -> torch.Tensor:
    lum = 0.2126 * image[..., 2] + 0.7152 * image[..., 1] + 0.0722 * image[..., 0]
    log_avg = torch.exp(torch.log(lum + epsilon).mean())
    key = base_key / log_avg
    scaled = image * key
    return (scaled / (1.0 + scaled))**gamma


def load_rgb_image(image_path: Path) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_array = cv2.imread(str(image_path), -1)
    rgb_array = img_array.astype(np.float32)
    rgb_array = rgb_array / max(np.max(rgb_array), 1.0)
    return torch.from_numpy(rgb_array).cuda()


def test_bilateral(image_path: Path, args):
    print(f"Loading image: {image_path}")
    input_rgb = load_rgb_image(image_path)
    height, width, channels = input_rgb.shape
    print(f"Image size: {width}x{height}x{channels}")

    print("Creating Bilateral algorithm...")
    workspace = create_bilateral(input_rgb.device, (width, height), args.sigma_s, args.sigma_r, args.detail)

    print(f"Parameters: sigma_s={args.sigma_s}, sigma_r={args.sigma_r}, detail={args.detail}")

    print("Applying bilateral filter to RGB image...")
    result_rgb = bilateral_rgb(workspace, input_rgb)

    if args.tonemap:
        input_rgb = reinhard(input_rgb)
        result_rgb = reinhard(result_rgb)

    input_display = (input_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    result_display = (result_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    combined = np.hstack([input_display, result_display])

    print("Showing results (original | filtered)...")
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", combined)
    while cv2.waitKey(1):
        pass
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Test bilateral grid filter on an image')
    parser.add_argument('image', type=Path, help='Input image path')
    parser.add_argument('--sigma_s', type=float, default=8.0, help='Spatial sigma (pixels)')
    parser.add_argument('--sigma_r', type=float, default=0.1, help='Range sigma (luminance, 0-1)')
    parser.add_argument('--detail', type=float, default=0.0, help='Detail amount (0 no change, <0 smooth, >0 boost)')
    parser.add_argument('--tonemap', action='store_true', help='Tonemap the output image for display')

    args = parser.parse_args()
    test_bilateral(args.image, args)


if __name__ == "__main__":
    main()


