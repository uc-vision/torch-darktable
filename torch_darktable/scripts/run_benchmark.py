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


    ppg_alg = td.create_ppg(bayer_input.device, (width, height), pattern)
    rcd_alg = td.create_rcd(bayer_input.device, (width, height), pattern)
    color_smooth_alg = td.create_postprocess(bayer_input.device, (width, height), pattern, color_smoothing_passes=3)
    green_eq_alg = td.create_postprocess(bayer_input.device, (width, height), pattern, green_eq_local=True, green_eq_global=True)
    
    laplacian_alg = td.create_laplacian(bayer_input.device, (width, height), params=LaplacianParams())
    
    bilateral_2x2 = td.create_bilateral(bayer_input.device, (width, height), sigma_s=2.0, sigma_r=0.2)
    bilateral_8x1 = td.create_bilateral(bayer_input.device, (width, height), sigma_s=8.0, sigma_r=0.1)

    print("=== Demosaic Algorithm Benchmarks ===")

    benchmark("PPG", ppg_alg.process, bayer_input, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("RCD", rcd_alg.process, bayer_input, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Bilinear 5x5", td.bilinear5x5_demosaic, bayer_input, pattern, warmup_iters=warmup_iters, bench_iters=bench_iters)

    print()

    print("=== Post-processing Benchmarks ===")
    benchmark("Color smooth", color_smooth_alg.process, rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Green eq", green_eq_alg.process, rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)

    print()

    print("=== Laplacian/Bilateral Benchmarks ===")

    mono_tensor = td.compute_luminance(rgb_tensor)
    benchmark("Laplacian", laplacian_alg.process, mono_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Bilateral Contrast", partial(bilateral_2x2.process_contrast, detail=0.2), mono_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Bilateral Contrast", partial(bilateral_8x1.process_contrast, detail=0.2), mono_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)

    benchmark("Bilateral Denoise RGB", partial(bilateral_2x2.process_denoise, amount=0.2), rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Bilateral Denoise RGB", partial(bilateral_8x1.process_denoise, amount=0.2), rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)



def main():
    parser = argparse.ArgumentParser(description='Benchmark demosaic algorithms and post-processing')
    parser.add_argument('image', type=Path, help='Input image path')
    parser.add_argument('--pattern', type=str, default='RGGB', choices=[p.name for p in BayerPattern],
                       help='Bayer pattern (default: RGGB)')
    parser.add_argument('--warmup-iters', type=int, default=5,
                       help='Number of warmup iterations (default: 5)')
    parser.add_argument('--bench-iters', type=int, default=1000,
                       help='Number of benchmark iterations (default: 50)')
    
    args = parser.parse_args()
    
    run_benchmark(args.image, BayerPattern[args.pattern], 
                 args.warmup_iters, args.bench_iters)


if __name__ == "__main__":
    main()
