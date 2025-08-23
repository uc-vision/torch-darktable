import torch
import argparse
from pathlib import Path
from typing import Callable

from torch_darktable import BayerPattern
from torch_darktable.utilities import load_image, rgb_to_bayer
from torch_darktable import ppg_demosaic, rcd_demosaic, postprocess_demosaic


def benchmark(func: Callable, *args, warmup_iters: int = 5, bench_iters: int = 50) -> float:
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

    return (1000.0 * bench_iters) / elapsed_ms




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

    print("=== Demosaic Algorithm Benchmarks ===")

    # Benchmark PPG using pre-created algorithm
    time_ms = benchmark(ppg_alg.process, bayer_input, warmup_iters=warmup_iters, bench_iters=bench_iters)
    print(f"PPG           : {time_ms:6.2f} images/sec")

    # Benchmark RCD using pre-created algorithm
    time_ms = benchmark(rcd_alg.process, bayer_input, warmup_iters=warmup_iters, bench_iters=bench_iters)
    print(f"RCD           : {time_ms:6.2f} images/sec")

    print()

    # Benchmark post-processing (use PPG result as input)
    print("=== Post-processing Benchmarks ===")
    ppg_result = ppg_alg.process(bayer_input)

    time_ms = benchmark(color_smooth_alg.process, ppg_result, warmup_iters=warmup_iters, bench_iters=bench_iters)
    print(f"Color smooth  : {time_ms:6.2f} images/sec")

    time_ms = benchmark(green_eq_alg.process, ppg_result, warmup_iters=warmup_iters, bench_iters=bench_iters)
    print(f"Green eq      : {time_ms:6.2f} images/sec")


def main():
    parser = argparse.ArgumentParser(description='Benchmark demosaic algorithms and post-processing')
    parser.add_argument('image', type=Path, help='Input image path')
    parser.add_argument('--pattern', type=str, default='RGGB', choices=[p.name for p in BayerPattern],
                       help='Bayer pattern (default: RGGB)')
    parser.add_argument('--warmup-iters', type=int, default=5,
                       help='Number of warmup iterations (default: 5)')
    parser.add_argument('--bench-iters', type=int, default=200,
                       help='Number of benchmark iterations (default: 50)')
    
    args = parser.parse_args()
    
    run_benchmark(args.image, BayerPattern[args.pattern], 
                 args.warmup_iters, args.bench_iters)


if __name__ == "__main__":
    main()
