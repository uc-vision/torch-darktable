

def create_laplacian(
    width: int,
    height: int,
    params: LaplacianParams | None = None):
    """
    Create a reusable Laplacian filter for local tone mapping.

    Args:
        params: LaplacianParams object (uses defaults if None)
        width: Image width that will be processed
        height: Image height that will be processed

    Returns:
        Laplacian workspace object that can be reused for multiple images
    """
    if params is None:
        params = LaplacianParams()

    print(f"Creating Laplacian filter for {width}x{height} with params {params}")

    assert width > 0 and height > 0, "Width and height must be positive"

    return demosaic_cuda.Laplacian(
        params.num_gamma, width, height, params.sigma, params.shadows, params.highlights, params.clarity)


def local_laplacian_rgb(
    image: torch.Tensor,
    laplacian) -> torch.Tensor:
    """
    Apply local Laplacian filtering to RGB image.

    This function:
    1. Extracts luminance from RGB using LAB color space
    2. Processes luminance with local Laplacian filter
    3. Reconstructs RGB with modified luminance

    Args:
        image: Input RGB image tensor of shape (H, W, 3)
               Must be on CUDA device and float32 dtype, values 0-1
        laplacian: Laplacian workspace object created by create_laplacian()

    Returns:
        Filtered RGB image of same shape and type as input
    """
    assert image.dim() == 3 and image.size(2) == 3, "Input must be 3D tensor (H, W, 3)"
    assert image.device.type == 'cuda', "Input must be on CUDA device"
    assert image.dtype == torch.float32, "Input must be float32 dtype"

    # Extract luminance
    luminance = demosaic_cuda.compute_luminance(image)

    # Ensure luminance is contiguous and properly aligned
    luminance = luminance.contiguous()

    # Process luminance
    processed_luminance = laplacian.process(luminance)

    # Reconstruct RGB with modified luminance
    return demosaic_cuda.modify_luminance(image, processed_luminance)




