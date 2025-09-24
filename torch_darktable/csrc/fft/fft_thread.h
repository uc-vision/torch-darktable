#pragma once

#include "common.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Helper function for swapping
template<typename T>
__device__ __forceinline__ void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}



// Butterfly operation helper
__device__ __forceinline__ void butterfly(Complex& a, Complex& b, const Complex& twiddle) {
    Complex temp = b * twiddle;
    b = a - temp;
    a = a + temp;
}

// Core FFT implementation
template<typename FFT>
__device__ __forceinline__ void fft_core(Complex row[FFT::N]) {
    constexpr int stages = log2_constexpr(FFT::N);
    
    // Bit-reverse permutation
    #pragma unroll
    for (int i = 0; i < FFT::N; i++) {
        int rev_i = __brev(i) >> (32 - stages);
        if (i < rev_i) swap(row[i], row[rev_i]);
    }
    
    // Radix-2 butterflies
    #pragma unroll
    for (int stage = 0; stage < stages; stage++) {
        int step = 1 << stage;
        int twiddle_stride = FFT::N >> (stage + 1);
        
        #pragma unroll
        for (int i = 0; i < FFT::N; i += 2 * step) {
            #pragma unroll
            for (int j = 0; j < step; j++) {
                Complex twiddle = FFT::twiddle(j * twiddle_stride);
                butterfly(row[i + j], row[i + j + step], twiddle);
            }
        }
    }
}



// Forward FFT
template<int N>
__device__ __forceinline__ void fft(Complex row[N]) {
    fft_core<FFT<N>>(row);
}

// Inverse FFT
template<int N>
__device__ __forceinline__ void ifft(Complex row[N]) {
    fft_core<IFFT<N>>(row);
    #pragma unroll
    for (int i = 0; i < N; i++) {
        row[i] = row[i] * (1.0f / N);
    }
}

template<int N>
__device__ __forceinline__ void transpose_rows(Complex row[N]) {
    auto warp = cg::tiled_partition<N>(cg::this_thread_block());
    int tid = warp.thread_rank();
    
    Complex temp[N];
    #pragma unroll
    for (int i = 0; i < N; i++) {
        temp[i] = row[i];
    }
    
    #pragma unroll
    for (int i = 0; i < N; i++) {
        row[i] = Complex(
            warp.shfl(temp[i].re, tid),
            warp.shfl(temp[i].im, tid)
        );
    }
}

template<int N>
__device__ __forceinline__ void fft_2d(Complex row[N]) {
    fft<N>(row);
    transpose_rows<N>(row);
    fft<N>(row);
}

template<int N>
__device__ __forceinline__ void ifft_2d(Complex row[N]) {
    ifft<N>(row);
    transpose_rows<N>(row);
    ifft<N>(row);
}

