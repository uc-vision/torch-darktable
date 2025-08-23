import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from torch_darktable import local_laplacian_rgb, create_laplacian, LaplacianParams
from torch_darktable import compute_luminance

def create_laplacian_algorithm(device, height, width, args):
    """Create Laplacian algorithm from args object."""
    params = LaplacianParams(
        num_gamma=args.num_gamma,
        sigma=args.sigma,
        shadows=args.shadows,
        highlights=args.highlights,
        clarity=args.clarity
    )
    return create_laplacian(device, (width, height), params=params), params


# def reinhard(image: torch.Tensor) -> torch.Tensor:
#     lum = compute_luminance(image)
#     log_lum = (lum + 1e-6).log().mean()
#     key = (lum.max() - log_lum) / (lum.max() - lum.min())
#     map_key = 0.3 + 0.7 * key**1.4

#     return image * (1.0 / (map_key + 1.0))-


def reinhard(image: torch.Tensor, epsilon=1e-4, base_key=0.18, gamma=0.75) -> torch.Tensor:
    lum = compute_luminance(image)
    log_avg = torch.exp(torch.log(lum + epsilon).mean())
    key = base_key / log_avg

    scaled = image * key

    tonemapped = (scaled / (1.0 + scaled))**gamma

    return tonemapped.clamp(0.0, 1.0)


def load_rgb_image(image_path: Path) -> torch.Tensor:
    """Load RGB image as tensor."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load as RGB
    img = Image.open(image_path).convert('RGB')
    rgb_array = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor (H, W, 3)
    tensor = torch.from_numpy(rgb_array).cuda()
    
    return tensor

def test_laplacian(image_path: Path, args):
    """Test local Laplacian filter on RGB image using LAB color space processing."""
    print(f"Loading image: {image_path}")

    # Load RGB image
    input_rgb = load_rgb_image(image_path)
    height, width, channels = input_rgb.shape

    print(f"Image size: {width}x{height}x{channels}")
    print(f"RGB range: {input_rgb.min().item():.3f} - {input_rgb.max().item():.3f}")

    # Create Laplacian algorithm from args
    print("Creating Laplacian algorithm...")
    workspace, params = create_laplacian_algorithm(input_rgb.device, height, width, args)

    print(f"Parameters: gamma={params.num_gamma}, sigma={params.sigma}, shadows={params.shadows}, highlights={params.highlights}, clarity={params.clarity}")

    if args.tonemap:
        input_rgb = reinhard(input_rgb)


    # Apply local Laplacian filter with RGB->LAB->RGB conversion
    print("Applying local Laplacian filter to RGB image...")
    result_rgb = local_laplacian_rgb(workspace, input_rgb)


    # Convert results to displayable format
    input_display = (input_rgb.cpu().numpy() * 255).astype(np.uint8)
    result_display = (result_rgb.cpu().numpy() * 255).astype(np.uint8)


    # Create side-by-side comparison
    combined = np.hstack([input_display, result_display])

    # Show results
    combined_img = Image.fromarray(combined)

    print("Showing results (original | filtered)...")
    combined_img.show()

    print(f"Output RGB range: {result_rgb.min().item():.3f} - {result_rgb.max().item():.3f}")


def main():
    parser = argparse.ArgumentParser(description='Test local Laplacian filter on an image')
    parser.add_argument('image', type=Path, help='Input image path')
    parser.add_argument('--num_gamma', type=int, default=6,
                       help='Number of gamma levels (4, 6, or 8)')
    parser.add_argument('--sigma', type=float, default=0.2,
                       help='Tone mapping parameter controlling transitions (default: 0.2)')
    parser.add_argument('--shadows', type=float, default=1.0,
                       help='Shadow enhancement, default: 1.0')
    parser.add_argument('--highlights', type=float, default=1.0,
                       help='Highlight compression, default: 1.0')
    parser.add_argument('--clarity', type=float, default=0.0,
                       help='Local contrast enhancement, default: 0.0')

    parser.add_argument('--tonemap', action='store_true',
                       help='Tonemap the output image')

    args = parser.parse_args()

    test_laplacian(args.image, args)


if __name__ == "__main__":
    main()
