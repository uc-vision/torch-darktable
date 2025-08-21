import torch
import ppg_demosaic_cuda
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


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


def test_ppg_demosaic(image_path: Path, filters: int = 0x94949494):
    """Test PPG demosaic on a real image."""
    print(f"Loading image: {image_path}")
    
    # Convert RGB image to Bayer pattern
    bayer_input = rgb_to_bayer(image_path)
    height, width = bayer_input.shape[:2]
    
    print(f"Image size: {width}x{height}")
    print(f"Input range: {bayer_input.min().item():.3f} - {bayer_input.max().item():.3f}")
    
    # Run demosaic
    print("Running PPG demosaic...")
    result = ppg_demosaic_cuda.ppg_demosaic(bayer_input, filters)
    
    print(f"Output shape: {result.shape}")
    
    # Check results
    r_channel = result[:, :, 0]
    g_channel = result[:, :, 1] 
    b_channel = result[:, :, 2]
    
    print(f"Red range: {r_channel.min().item():.3f} - {r_channel.max().item():.3f}")
    print(f"Green range: {g_channel.min().item():.3f} - {g_channel.max().item():.3f}")
    print(f"Blue range: {b_channel.min().item():.3f} - {b_channel.max().item():.3f}")
    
    # Save result
    output_path = image_path.with_suffix('.demosaiced.png')
    
    # Convert back to PIL and save
    rgb = result[:, :, :3].clamp(0, 1)
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(rgb_np).save(output_path)
    
    print(f"Saved demosaiced image: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test PPG demosaic on an image')
    parser.add_argument('image', type=Path, help='Input image path')
    parser.add_argument('--filters', type=lambda x: int(x, 16), default=0x94949494,
                       help='Bayer pattern (hex, default: 0x94949494 for RGGB)')
    
    args = parser.parse_args()
    
    try:
        test_ppg_demosaic(args.image, args.filters)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())