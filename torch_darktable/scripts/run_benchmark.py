from functools import partial
import torch
import argparse
from pathlib import Path
from typing import Callable

from torch_darktable import BayerPattern
from torch_darktable.local_contrast import LaplacianParams
from torch_darktable.utilities import load_image, rgb_to_bayer
import torch_darktable as td


def benchmark(name: str, func: Callable, *args, warmup_iters: int = 5, bench_iters: int = 50) -> float:
    # Warmup
    for _ in range(warmup_iters):
        func(*args)

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    stream = torch.cuda.current_stream()

    start_event.record(stream=stream)
    for _ in range(bench_iters):
        func(*args)
    end_event.record(stream=stream)

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    rate = (1000.0 * bench_iters) / elapsed_ms
    print(f"{name}: {bench_iters} iterations in {elapsed_ms:.3f}ms at {rate:.1f} iters/sec")
    return rate


def run_benchmark(image_path: Path, pattern: BayerPattern, warmup_iters: int = 5,
                 bench_iters: int = 50):

    torch.set_grad_enabled(False)

    print(f"Loading image: {image_path}")

    # Load and convert image (not timed)
    rgb_tensor = load_image(image_path)
    bayer_input = rgb_to_bayer(rgb_tensor)




    height, width = bayer_input.shape[:2]

    print(f"Image size: {width}x{height}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")
    print(f"Pattern: {pattern.name}")
    print()


    ppg = td.create_ppg(bayer_input.device, (width, height), pattern)
    rcd = td.create_rcd(bayer_input.device, (width, height), pattern)
    color_smooth = td.create_postprocess(bayer_input.device, (width, height), pattern, color_smoothing_passes=3)
    green_eq = td.create_postprocess(bayer_input.device, (width, height), pattern, green_eq_local=True, green_eq_global=True)
    
    laplacian = td.create_laplacian(bayer_input.device, (width, height), params=LaplacianParams())
    
    bilateral_2x2 = td.create_bilateral(bayer_input.device, (width, height), sigma_s=2.0, sigma_r=0.2)
    bilateral_8x1 = td.create_bilateral(bayer_input.device, (width, height), sigma_s=8.0, sigma_r=0.1)
    
    
    wiener32x2 = td.create_wiener(bayer_input.device, (width, height), overlap=2, tile_size=32)
    wiener32x4 = td.create_wiener(bayer_input.device, (width, height), overlap=4, tile_size=32)
    wiener32x2_gray = td.create_wiener(bayer_input.device, (width, height), overlap=2, tile_size=32, channels=1)


    print("=== Denoise Benchmarks ===")

    benchmark("Wiener 32x2", partial(wiener32x2.process, noise=0.05), rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Wiener 32x4", partial(wiener32x4.process, noise=0.05), rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)

    benchmark("Wiener 32x2 Gray", partial(wiener32x2_gray.process_luminance, noise=0.05), rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)


    benchmark("Estimate Noise", td.estimate_channel_noise, rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)

    print("=== Demosaic Algorithm Benchmarks ===")

    benchmark("PPG", ppg.process, bayer_input, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("RCD", rcd.process, bayer_input, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Bilinear 5x5", td.bilinear5x5_demosaic, bayer_input, pattern, warmup_iters=warmup_iters, bench_iters=bench_iters)

    print()

    print("=== Post-processing Benchmarks ===")
    benchmark("Color smooth", color_smooth.process, rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Green eq", green_eq.process, rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)

    print()

    print("=== Laplacian/Bilateral Benchmarks ===")

    mono_tensor = td.compute_luminance(rgb_tensor)
    benchmark("Laplacian", laplacian.process, mono_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Bilateral 2x2", partial(bilateral_2x2.process, detail=0.2), mono_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Bilateral 8x1", partial(bilateral_8x1.process, detail=0.2), mono_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)



def main():
    parser = argparse.ArgumentParser(description='Benchmark demosaic algorithms and post-processing')
    parser.add_argument('image', type=Path, help='Input image path')

    parser.add_argument('--pattern', type=str, default='RGGB', choices=[p.name for p in BayerPattern],
                       help='Bayer pattern (default: RGGB)')
    parser.add_argument('--warmup-iters', type=int, default=5,
                       help='Number of warmup iterations (default: 5)')
    parser.add_argument('--bench-iters', type=int, default=100,
                       help='Number of benchmark iterations (default: 50)')
    
    args = parser.parse_args()
    
    run_benchmark(args.image, BayerPattern[args.pattern], 
                 args.warmup_iters, args.bench_iters)


if __name__ == "__main__":
    main()
