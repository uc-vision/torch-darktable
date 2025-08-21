import torch

import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from cuda_demosaic import ppg_demosaic, rcd_demosaic, BayerPattern

def rgb_to_bayer(image_path: Path) -> torch.Tensor:
    """Load RGB image and convert to Bayer pattern."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load as RGB
    img = Image.open(image_path).convert('RGB')
    rgb_array = np.array(img, dtype=np.float32) / 255.0
    
    height, width = rgb_array.shape[:2]
    bayer = np.zeros((height, width), dtype=np.float32)
    
    # Create RGGB Bayer pattern using numpy indexing
    bayer[0::2, 0::2] = rgb_array[0::2, 0::2, 0]  # R (even rows, even cols)
    bayer[0::2, 1::2] = rgb_array[0::2, 1::2, 1]  # G (even rows, odd cols)
    bayer[1::2, 0::2] = rgb_array[1::2, 0::2, 1]  # G (odd rows, even cols)  
    bayer[1::2, 1::2] = rgb_array[1::2, 1::2, 2]  # B (odd rows, odd cols)
    
    # Convert to tensor (H, W, 1)
    tensor = torch.from_numpy(bayer).unsqueeze(-1).cuda()
    
    return tensor


def test_demosaic(image_path: Path, pattern: BayerPattern, algorithm: str = "ppg", 
                 median_threshold: float | None = None, input_scale: float = 1.0, output_scale: float = 1.0):
    """Test demosaic on a real image."""
    print(f"Loading image: {image_path}")
    
    # Convert RGB image to Bayer pattern
    bayer_input = rgb_to_bayer(image_path)
    height, width = bayer_input.shape[:2]
    
    print(f"Image size: {width}x{height}")
    print(f"Input range: {bayer_input.min().item():.3f} - {bayer_input.max().item():.3f}")
    
    # Run demosaic
    if algorithm.lower() == "ppg":
        print("Running PPG demosaic...")
        result = ppg_demosaic(bayer_input, pattern, median_threshold=median_threshold)
    elif algorithm.lower() == "rcd":
        print("Running RCD demosaic...")
        result = rcd_demosaic(bayer_input, pattern, input_scale=input_scale, output_scale=output_scale)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'ppg' or 'rcd'")
    
  
    rgb = result[:, :, :3].clamp(0, 1)
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(rgb_np)
    image.show()
    
    


def main():
    parser = argparse.ArgumentParser(description='Test demosaic algorithms on an image')
    parser.add_argument('image', type=Path, help='Input image path')
    parser.add_argument('--pattern', type=str, default='RGGB', choices=[pattern.name for pattern in BayerPattern],
                       help='Bayer pattern')
    parser.add_argument('--algorithm', type=str, default='ppg', choices=['ppg', 'rcd'],
                       help='Demosaic algorithm to use')
    parser.add_argument('--median_threshold', type=float, default=None,
                       help='Median threshold (PPG only)')

    parser.add_argument('--input_scale', type=float, default=1.0,
                       help='Input scaling factor (RCD only)')
    parser.add_argument('--output_scale', type=float, default=1.0,
                       help='Output scaling factor (RCD only)')
    
    args = parser.parse_args()
    
    try:
        test_demosaic(args.image, BayerPattern[args.pattern], args.algorithm,
                     args.median_threshold, args.input_scale, args.output_scale)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())