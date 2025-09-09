import torch
import argparse
from pathlib import Path
from typing import Callable

from torch_darktable import BayerPattern
from torch_darktable.utilities import load_image, rgb_to_bayer
from torch_darktable import ppg_demosaic, rcd_demosaic, postprocess_demosaic, create_laplacian, compute_luminance
from torch_darktable import create_bilateral


def benchmark(name: str, func: Callable, *args, warmup_iters: int = 5, bench_iters: int = 50) -> float:
    # Warmup
    for _ in range(warmup_iters):
        func(*args)

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(bench_iters):
        func(*args)
    end_event.record()

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


    ppg_alg = ppg_demosaic(bayer_input.device, (width, height), pattern)
    rcd_alg = rcd_demosaic(bayer_input.device, (width, height), pattern)
    color_smooth_alg = postprocess_demosaic(bayer_input.device, (width, height), pattern, color_smoothing_passes=3)
    green_eq_alg = postprocess_demosaic(bayer_input.device, (width, height), pattern,
                                    green_eq_local=True, green_eq_global=True)

    laplacian_alg = create_laplacian(bayer_input.device, (width, height))
    bilateral_alg = create_bilateral(bayer_input.device, (width, height), sigma_s=2.0, sigma_r=0.2, detail=0.2)

    print("=== Demosaic Algorithm Benchmarks ===")

    benchmark("PPG", ppg_alg.process, bayer_input, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("RCD", rcd_alg.process, bayer_input, warmup_iters=warmup_iters, bench_iters=bench_iters)

    print()

    print("=== Post-processing Benchmarks ===")
    benchmark("Color smooth", color_smooth_alg.process, rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Green eq", green_eq_alg.process, rgb_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)

    print()

    print("=== Laplacian/Bilateral Benchmarks ===")

    mono_tensor = compute_luminance(rgb_tensor)
    benchmark("Laplacian", laplacian_alg.process, mono_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)
    benchmark("Bilateral", bilateral_alg.process, mono_tensor, warmup_iters=warmup_iters, bench_iters=bench_iters)



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
