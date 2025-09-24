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

// In-place transpose for NxN array stored in row-major order
template<int N>
__device__ __forceinline__ void transpose_inplace(Complex data[N*N]) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = i + 1; j < N; j++) {
            swap(data[i * N + j], data[j * N + i]);
        }
    }
}

// Butterfly operation helper
__device__ __forceinline__ void butterfly(Complex& a, Complex& b, const Complex& twiddle) {
    Complex temp = b * twiddle;
    b = a - temp;
    a = a + temp;
}

// Core FFT implementation
template<int N>
__device__ __forceinline__ void fft_core(Complex data[N], const Complex* twiddles) {
    constexpr int stages = log2_constexpr(N);
    
    // Bit-reverse permutation
    #pragma unroll
    for (int i = 0; i < N; i++) {
        int rev_i = __brev(i) >> (32 - stages);
        if (i < rev_i) swap(data[i], data[rev_i]);
    }
    
    // Radix-2 butterflies
    #pragma unroll
    for (int stage = 0; stage < stages; stage++) {
        int step = 1 << stage;
        int twiddle_stride = N >> (stage + 1);
        
        #pragma unroll
        for (int i = 0; i < N; i += 2 * step) {
            #pragma unroll
            for (int j = 0; j < step; j++) {
                butterfly(data[i + j], data[i + j + step], twiddles[j * twiddle_stride]);
            }
        }
    }
}


// Forward FFT
template<int N>
__device__ __forceinline__ void fft(Complex data[N]) {
    fft_core<N>(data, get_fft_twiddles<N>());
}

// Inverse FFT
template<int N>
__device__ __forceinline__ void ifft(Complex data[N]) {
    fft_core<N>(data, get_ifft_twiddles<N>());
    #pragma unroll
    for (int i = 0; i < N; i++) {
        data[i] = data[i] * (1.0f / N);
    }
}

template<int N>
__device__ __forceinline__ void transpose_rows(Complex my_row[N]) {
    auto warp = cg::this_thread_block();
    int tid = warp.thread_rank();
    
    Complex temp[N];
    #pragma unroll
    for (int i = 0; i < N; i++) {
        temp[i] = my_row[i];
    }
    
    #pragma unroll
    for (int i = 0; i < N; i++) {
        my_row[i] = Complex(
            warp.shfl(temp[tid].re, i, N),
            warp.shfl(temp[tid].im, i, N)
        );
    }
}

template<int N>
__device__ __forceinline__ void fft_2d_rows(Complex my_row[N]) {
    fft<N>(my_row);
    transpose_rows<N>(my_row);
    fft<N>(my_row);
}

template<int N>
__device__ __forceinline__ void ifft_2d_rows(Complex my_row[N]) {
    ifft<N>(my_row);
    transpose_rows<N>(my_row);
    ifft<N>(my_row);
}

