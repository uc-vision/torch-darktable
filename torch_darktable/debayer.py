"""Bayer demosaicing algorithms and utilities."""

import torch
from enum import IntEnum, Enum
from .extension import extension


class BayerPattern(IntEnum):
    """Bayer pattern enumeration for demosaicing."""
    RGGB = 0x94949494
    BGGR = 0x16161616
    GRBG = 0x61616161
    GBRG = 0x49494949


class Packed12Format(Enum):
    STANDARD = 0
    IDS = 1


def create_ppg(
    device: torch.device,
    image_size: tuple[int, int],
    bayer_pattern: BayerPattern,
    median_threshold: float = 0.0
) -> extension.PPG:
    """
    Create a PPG demosaic object.
    """
    width, height = image_size
    return extension.PPG(device, width, height, int(bayer_pattern), median_threshold)


def create_rcd(
    device: torch.device,
    image_size: tuple[int, int],
    bayer_pattern: BayerPattern,
    input_scale: float = 1.0,
    output_scale: float = 1.0
) -> extension.RCD:
    """
    Create an RCD demosaic object.
    """
    width, height = image_size
    return extension.RCD(device, width, height, int(bayer_pattern), input_scale, output_scale)


def create_postprocess(
    device: torch.device,
    image_size: tuple[int, int],
    bayer_pattern: BayerPattern,
    color_smoothing_passes: int = 0,
    green_eq_local: bool = False,
    green_eq_global: bool = False,
    green_eq_threshold: float = 0.04
) -> extension.PostProcess:
    """
    Create a post-process object for demosaiced images.
    """
    width, height = image_size
    return extension.PostProcess(
        device,
        width,
        height,
        int(bayer_pattern),
        color_smoothing_passes,
        green_eq_local,
        green_eq_global,
        green_eq_threshold,
    )


# 12-bit packing/unpacking functions
def encode12(image: torch.Tensor, format_type: Packed12Format = Packed12Format.STANDARD) -> torch.Tensor:
    """
    Encode image data to 12-bit packed format.
    
    Args:
        image: Input image tensor (uint16 or float32)
        format_type: "standard" or "ids" format
        
    Returns:
        Packed 12-bit data as uint8 tensor
    """
    ids = (format_type is Packed12Format.IDS)
    if image.dtype == torch.uint16:
        return extension.encode12_u16(image, ids_format=ids)
    elif image.dtype == torch.float32:
        return extension.encode12_float(image, ids_format=ids)
    else:
        raise ValueError(f"Unsupported input dtype: {image.dtype}")


def decode12(packed_data: torch.Tensor, output_dtype: torch.dtype = torch.float32, format_type: Packed12Format = Packed12Format.STANDARD) -> torch.Tensor:
    """
    Decode 12-bit packed data to image format.
    
    Args:
        packed_data: Packed 12-bit data as uint8 tensor
        output_dtype: torch.float32, torch.float16, or torch.uint16
        format_type: "standard" or "ids" format
        
    Returns:
        Decoded image tensor
    """
    ids = (format_type is Packed12Format.IDS)
    if output_dtype == torch.float32:
        return extension.decode12_float(packed_data, ids_format=ids)
    elif output_dtype == torch.float16:
        return extension.decode12_half(packed_data, ids_format=ids)
    elif output_dtype == torch.uint16:
        return extension.decode12_u16(packed_data, ids_format=ids)
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")


# Direct extension function access for individual functions
encode12_u16 = extension.encode12_u16
encode12_float = extension.encode12_float
decode12_float = extension.decode12_float
decode12_half = extension.decode12_half
decode12_u16 = extension.decode12_u16


__all__ = [
    "BayerPattern", "Packed12Format",
    "create_ppg", "create_rcd", "create_postprocess",
    "encode12", "decode12",
    "encode12_u16", "encode12_float", "decode12_float", "decode12_half", "decode12_u16",
]
