#pragma once

#include <cuda_runtime.h>
#include <cmath>

struct Complex {
    float re, im;
    
    __device__ __host__ constexpr Complex() : re(0.0f), im(0.0f) {}
    __device__ __host__ constexpr Complex(float real, float imag) : re(real), im(imag) {}
    
    __device__ __host__ constexpr Complex operator+(const Complex& other) const {
        return Complex(re + other.re, im + other.im);
    }
    
    __device__ __host__ constexpr Complex operator-(const Complex& other) const {
        return Complex(re - other.re, im - other.im);
    }
    
    __device__ __host__ constexpr Complex operator*(const Complex& other) const {
        return Complex(re * other.re - im * other.im, re * other.im + im * other.re);
    }
    
    __device__ __host__ constexpr Complex operator*(float scalar) const {
        return Complex(re * scalar, im * scalar);
    }
    
    __device__ __host__ constexpr Complex operator+(float scalar) const {
        return Complex(re + scalar, im);
    }
    
    __device__ __host__ constexpr Complex operator-(float scalar) const {
        return Complex(re - scalar, im);
    }
    
    __device__ __host__ constexpr float magnitude_squared() const {
        return re * re + im * im;
    }
};

// Scalar operations from the left
__device__ __host__ constexpr Complex operator*(float scalar, const Complex& c) {
    return Complex(scalar * c.re, scalar * c.im);
}

__device__ __host__ constexpr Complex operator+(float scalar, const Complex& c) {
    return Complex(scalar + c.re, c.im);
}

__device__ __host__ constexpr Complex operator-(float scalar, const Complex& c) {
    return Complex(scalar - c.re, -c.im);
}

// Precomputed twiddle factors for 16-point FFT
__device__ constexpr Complex fft_twiddles_16[8] = {
    {1.0f, 0.0f},                    // k=0
    {0.9238795325f, -0.3826834324f}, // k=1
    {0.7071067812f, -0.7071067812f}, // k=2
    {0.3826834324f, -0.9238795325f}, // k=3
    {0.0f, -1.0f},                   // k=4
    {-0.3826834324f, -0.9238795325f}, // k=5
    {-0.7071067812f, -0.7071067812f}, // k=6
    {-0.9238795325f, -0.3826834324f}  // k=7
};

__device__ constexpr Complex ifft_twiddles_16[8] = {
    {1.0f, 0.0f},                    // k=0
    {0.9238795325f, 0.3826834324f},  // k=1
    {0.7071067812f, 0.7071067812f},  // k=2
    {0.3826834324f, 0.9238795325f},  // k=3
    {0.0f, 1.0f},                    // k=4
    {-0.3826834324f, 0.9238795325f}, // k=5
    {-0.7071067812f, 0.7071067812f}, // k=6
    {-0.9238795325f, 0.3826834324f}  // k=7
};

// Precomputed twiddle factors for 32-point FFT
__device__ constexpr Complex fft_twiddles_32[16] = {
    {1.0f, 0.0f},                    // k=0
    {0.9807852804f, -0.1950903220f}, // k=1  
    {0.9238795325f, -0.3826834324f}, // k=2
    {0.8314696123f, -0.5555702330f}, // k=3
    {0.7071067812f, -0.7071067812f}, // k=4
    {0.5555702330f, -0.8314696123f}, // k=5
    {0.3826834324f, -0.9238795325f}, // k=6
    {0.1950903220f, -0.9807852804f}, // k=7
    {0.0f, -1.0f},                   // k=8
    {-0.1950903220f, -0.9807852804f}, // k=9
    {-0.3826834324f, -0.9238795325f}, // k=10
    {-0.5555702330f, -0.8314696123f}, // k=11
    {-0.7071067812f, -0.7071067812f}, // k=12
    {-0.8314696123f, -0.5555702330f}, // k=13
    {-0.9238795325f, -0.3826834324f}, // k=14
    {-0.9807852804f, -0.1950903220f}  // k=15
};

__device__ constexpr Complex ifft_twiddles_32[16] = {
    {1.0f, 0.0f},                    // k=0
    {0.9807852804f, 0.1950903220f},  // k=1
    {0.9238795325f, 0.3826834324f},  // k=2
    {0.8314696123f, 0.5555702330f},  // k=3
    {0.7071067812f, 0.7071067812f},  // k=4
    {0.5555702330f, 0.8314696123f},  // k=5
    {0.3826834324f, 0.9238795325f},  // k=6
    {0.1950903220f, 0.9807852804f},  // k=7
    {0.0f, 1.0f},                    // k=8
    {-0.1950903220f, 0.9807852804f}, // k=9
    {-0.3826834324f, 0.9238795325f}, // k=10
    {-0.5555702330f, 0.8314696123f}, // k=11
    {-0.7071067812f, 0.7071067812f}, // k=12
    {-0.8314696123f, 0.5555702330f}, // k=13
    {-0.9238795325f, 0.3826834324f}, // k=14
    {-0.9807852804f, 0.1950903220f}  // k=15
};

// Compile-time log2 calculation
constexpr int log2_constexpr(int n) {
    return (n <= 1) ? 0 : 1 + log2_constexpr(n / 2);
}

// Twiddle factor selection
template<int N>
__device__ __forceinline__ constexpr const Complex* get_fft_twiddles();

template<>
__device__ __forceinline__ constexpr const Complex* get_fft_twiddles<16>() {
    return fft_twiddles_16;
}

template<>
__device__ __forceinline__ constexpr const Complex* get_fft_twiddles<32>() {
    return fft_twiddles_32;
}

template<int N>
__device__ __forceinline__ constexpr const Complex* get_ifft_twiddles();

template<>
__device__ __forceinline__ constexpr const Complex* get_ifft_twiddles<16>() {
    return ifft_twiddles_16;
}

template<>
__device__ __forceinline__ constexpr const Complex* get_ifft_twiddles<32>() {
    return ifft_twiddles_32;
}
