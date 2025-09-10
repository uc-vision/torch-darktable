import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from torch_darktable import create_ppg, create_rcd, create_postprocess, BayerPattern
from torch_darktable.utilities import load_image, rgb_to_bayer


def create_debayer_algorithm(device, width, height, pattern, args):
    """Create debayer algorithm from args object (handles PPG, RCD, and postprocessing)."""
    if args.algorithm.lower() == "ppg":
        return create_ppg(device, (width, height), pattern, median_threshold=args.median_threshold)
    elif args.algorithm.lower() == "rcd":
        return create_rcd(device, (width, height), pattern, input_scale=args.input_scale, output_scale=args.output_scale)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}. Choose 'ppg' or 'rcd'")


def create_postprocess_algorithm(device, width, height, pattern, args):
    """Create post-processing algorithm from args object."""
    return create_postprocess(device, (width, height), pattern, args.color_smoothing_passes,
                              args.green_eq_local, args.green_eq_global, args.green_eq_threshold)


def test_demosaic(image_path: Path, pattern: BayerPattern, args):
    """Test demosaic on a real image."""
    print(f"Loading image: {image_path}")
    
    # Load and convert RGB image to Bayer pattern
    rgb_tensor = load_image(image_path)
    bayer_input = rgb_to_bayer(rgb_tensor)
    height, width = bayer_input.shape[:2]
    device = rgb_tensor.device
    
    print(f"Image size: {width}x{height}")
    print(f"Input range: {bayer_input.min().item():.3f} - {bayer_input.max().item():.3f}")
    
    # Create demosaic algorithm
    print(f"Creating {args.algorithm.upper()} demosaic algorithm...")
    debayer_alg = create_debayer_algorithm(device, width, height, pattern, args)
    result = debayer_alg.process(bayer_input)

    # Apply post-processing if requested
    if args.color_smoothing_passes > 0 or args.green_eq_local or args.green_eq_global:
        print(f"Applying post-processing: smoothing={args.color_smoothing_passes}, green_eq_local={args.green_eq_local}, green_eq_global={args.green_eq_global}")

        postprocess_alg = create_postprocess_algorithm(device, width, height, pattern, args)
        result = postprocess_alg.process(result)
  
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
    parser.add_argument('--color_smoothing_passes', type=int, default=3,
                       help='Number of color smoothing passes (0 to disable)')
    parser.add_argument('--green_eq_local', action='store_true',
                       help='Enable local green equilibration')
    parser.add_argument('--green_eq_global', action='store_true',
                       help='Enable global green equilibration')
    parser.add_argument('--green_eq_threshold', type=float, default=0.01,
                       help='Green equilibration threshold')
    
    args = parser.parse_args()
    
    try:
        test_demosaic(args.image, BayerPattern[args.pattern], args)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())