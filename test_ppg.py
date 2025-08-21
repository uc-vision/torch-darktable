#!/usr/bin/env python3
"""
Test script for PPG Demosaic implementation
"""

import torch
import numpy as np
import time
from ppg_demosaic import ppg_demosaic, PPGDemosaic

def create_synthetic_bayer(height, width, pattern='RGGB'):
    """Create a synthetic Bayer pattern image for testing"""
    
    # Create base image with some structure
    x = torch.linspace(0, 1, width, device='cuda')
    y = torch.linspace(0, 1, height, device='cuda')
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create synthetic RGB channels
    r_channel = 0.8 * torch.sin(X * 10) * torch.cos(Y * 8) + 0.2
    g_channel = 0.6 * torch.cos(X * 8) * torch.sin(Y * 10) + 0.4  
    b_channel = 0.7 * torch.sin(X * 6) * torch.sin(Y * 6) + 0.3
    
    # Apply Bayer pattern sampling
    bayer = torch.zeros(height, width, device='cuda')
    
    if pattern == 'RGGB':
        # R G
        # G B
        bayer[0::2, 0::2] = r_channel[0::2, 0::2]  # Red
        bayer[0::2, 1::2] = g_channel[0::2, 1::2]  # Green  
        bayer[1::2, 0::2] = g_channel[1::2, 0::2]  # Green
        bayer[1::2, 1::2] = b_channel[1::2, 1::2]  # Blue
    elif pattern == 'BGGR':
        # B G
        # G R  
        bayer[0::2, 0::2] = b_channel[0::2, 0::2]  # Blue
        bayer[0::2, 1::2] = g_channel[0::2, 1::2]  # Green
        bayer[1::2, 0::2] = g_channel[1::2, 0::2]  # Green  
        bayer[1::2, 1::2] = r_channel[1::2, 1::2]  # Red
    elif pattern == 'GRBG':
        # G R
        # B G
        bayer[0::2, 0::2] = g_channel[0::2, 0::2]  # Green
        bayer[0::2, 1::2] = r_channel[0::2, 1::2]  # Red
        bayer[1::2, 0::2] = b_channel[1::2, 0::2]  # Blue
        bayer[1::2, 1::2] = g_channel[1::2, 1::2]  # Green
    elif pattern == 'GBRG':
        # G B  
        # R G
        bayer[0::2, 0::2] = g_channel[0::2, 0::2]  # Green
        bayer[0::2, 1::2] = b_channel[0::2, 1::2]  # Blue
        bayer[1::2, 0::2] = r_channel[1::2, 0::2]  # Red
        bayer[1::2, 1::2] = g_channel[1::2, 1::2]  # Green
    
    return bayer.clamp(0, 1)

def test_basic_functionality():
    """Test basic demosaicing functionality"""
    print("Testing basic functionality...")
    
    # Test different image sizes
    sizes = [(256, 256), (512, 768), (1024, 1536)]
    patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    
    for height, width in sizes:
        for pattern in patterns:
            print(f"  Testing {height}x{width} {pattern}...")
            
            # Create synthetic Bayer image
            bayer = create_synthetic_bayer(height, width, pattern)
            
            # Demosaic
            start_time = time.time()
            rgb = ppg_demosaic(bayer, pattern)
            end_time = time.time()
            
            # Check output
            assert rgb.shape == (height, width, 3), f"Wrong output shape: {rgb.shape}"
            assert rgb.device.type == 'cuda', "Output should be on CUDA"
            assert rgb.dtype == torch.float32, "Output should be float32"
            assert torch.all(rgb >= 0), "Output should be non-negative"
            
            print(f"    {(end_time - start_time)*1000:.1f}ms")
    
    print("✓ Basic functionality tests passed")

def test_median_filtering():
    """Test pre-median filtering"""
    print("Testing median filtering...")
    
    height, width = 512, 512
    bayer = create_synthetic_bayer(height, width, 'RGGB')
    
    # Add some noise
    noise = 0.05 * torch.randn_like(bayer)
    noisy_bayer = bayer + noise
    
    # Test different threshold values
    thresholds = [0.0, 0.01, 0.05, 0.1]
    
    demosaicer = PPGDemosaic()
    
    for threshold in thresholds:
        print(f"  Testing threshold {threshold}...")
        
        start_time = time.time()
        rgb = demosaicer.demosaic(noisy_bayer, 'RGGB', median_threshold=threshold)
        end_time = time.time()
        
        assert rgb.shape == (height, width, 3), "Wrong output shape"
        print(f"    {(end_time - start_time)*1000:.1f}ms")
    
    print("✓ Median filtering tests passed")

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    demosaicer = PPGDemosaic()
    
    # Test CPU tensor (should fail)
    try:
        cpu_tensor = torch.rand(256, 256)
        demosaicer.demosaic(cpu_tensor, 'RGGB')
        assert False, "Should have failed with CPU tensor"
    except ValueError as e:
        assert "CUDA device" in str(e)
        print("  ✓ CPU tensor error handling works")
    
    # Test wrong dtype
    try:
        int_tensor = torch.randint(0, 255, (256, 256), device='cuda', dtype=torch.uint8)
        result = demosaicer.demosaic(int_tensor, 'RGGB')  # Should auto-convert
        assert result.dtype == torch.float32
        print("  ✓ Dtype conversion works")
    except Exception as e:
        print(f"  ! Dtype test failed: {e}")
    
    # Test invalid Bayer pattern
    try:
        bayer = torch.rand(256, 256, device='cuda')
        demosaicer.demosaic(bayer, 'INVALID')
        assert False, "Should have failed with invalid pattern"
    except ValueError as e:
        assert "Unknown bayer pattern" in str(e)
        print("  ✓ Invalid pattern error handling works")
    
    # Test wrong dimensions
    try:
        wrong_shape = torch.rand(256, 256, 3, device='cuda')  # 3 channels
        demosaicer.demosaic(wrong_shape, 'RGGB')
        assert False, "Should have failed with wrong shape"
    except ValueError as e:
        assert "shape" in str(e)
        print("  ✓ Wrong shape error handling works")
    
    print("✓ Error handling tests passed")

def benchmark_performance():
    """Benchmark performance on different image sizes"""
    print("Benchmarking performance...")
    
    sizes = [
        (1024, 1024),     # 1MP
        (2048, 1536),     # 3MP  
        (2560, 1920),     # 5MP
        (3264, 2448),     # 8MP
        (4032, 3024),     # 12MP
    ]
    
    demosaicer = PPGDemosaic()
    
    for height, width in sizes:
        megapixels = (height * width) / 1e6
        print(f"  {height}x{width} ({megapixels:.1f}MP)...")
        
        # Create test image
        bayer = create_synthetic_bayer(height, width, 'RGGB')
        
        # Warmup
        for _ in range(3):
            _ = demosaicer.demosaic(bayer, 'RGGB')
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        n_runs = 10
        for _ in range(n_runs):
            rgb = demosaicer.demosaic(bayer, 'RGGB')
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs
        throughput = megapixels / avg_time
        
        print(f"    {avg_time*1000:.1f}ms avg ({throughput:.1f} MP/s)")
    
    print("✓ Performance benchmark completed")

def main():
    """Run all tests"""
    if not torch.cuda.is_available():
        print("CUDA not available - tests require GPU")
        return
    
    print("PPG Demosaic Test Suite")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        print()
        
        test_median_filtering()
        print()
        
        test_error_handling()
        print()
        
        benchmark_performance()
        print()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
