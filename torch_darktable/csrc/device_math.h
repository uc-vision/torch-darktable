// Vector arithmetic operators for int2
__device__ __host__ inline int2 operator+(const int2& a, const int2& b) {
    return make_int2(a.x + b.x, a.y + b.y);
}

__device__ __host__ inline int2 operator-(const int2& a, const int2& b) {
    return make_int2(a.x - b.x, a.y - b.y);
}

__device__ __host__ inline int2 operator-(const int2& a, int scalar) {
    return make_int2(a.x - scalar, a.y - scalar);
}

__device__ __host__ inline int2 operator+(const int2& a, int scalar) {
    return make_int2(a.x + scalar, a.y + scalar);
}

__device__ __host__ inline int2 operator*(int scalar, const int2& a) {
    return make_int2(scalar * a.x, scalar * a.y);
}

__device__ __host__ inline int2 operator/(const int2& a, int scalar) {
    return make_int2(a.x / scalar, a.y / scalar);
}

__device__ __forceinline__ float clipf(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ __host__ inline int clamp(int x, int low, int high) {
    return min(max(x, low), high);
}

__device__ __forceinline__ int2 get_thread_pos() {
    return make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

// Inline function instead of macro for better type safety and debugging
__device__ __forceinline__ int fc(int row, int col, uint32_t filters) {
    return (filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1)) & 3;
}


// Helper functions
__device__ __forceinline__ float fsquare(float x) {
    return x * x;
}



__device__ __forceinline__ float dtcl_sqrt(float x) {
    return sqrtf(x);
}

// Mix function equivalent
__device__ __forceinline__ float mix(float a, float b, float t) {
    return (1.0f - t) * a + t * b;
}



// Forward declaration of shared function from ppg_kernels.cu
__global__ void border_interpolate_kernel(float* input, float3* output, int width, int height, uint32_t filters, int border);
