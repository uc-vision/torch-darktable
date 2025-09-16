#pragma once

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

__device__ __host__ inline int2 operator*(const int2& a, int scalar) {
    return make_int2(a.x * scalar, a.y * scalar);
}

__device__ __host__ inline int2 operator/(const int2& a, int scalar) {
    return make_int2(a.x / scalar, a.y / scalar);
}

__device__ __forceinline__ float clipf(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

// Clamp float to [lo, hi]
__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __host__ inline int clamp(int x, int low, int high) {
    return min(max(x, low), high);
}

__device__ __forceinline__ int2 get_thread_pos() {
    return make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

// More descriptive alias for pixel indexing
__device__ __forceinline__ int2 pixel_index() {
    return get_thread_pos();
}

// Strided pixel indexing for sparse sampling
__device__ __forceinline__ int2 pixel_index_strided(int stride) {
    int2 pos = pixel_index();
    return make_int2(pos.x * stride, pos.y * stride);
}

// 1D thread indexing for linear arrays
__device__ __forceinline__ int thread_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
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

// 3x3 matrix type
struct float3x3 {
    float m[9];
    
    __device__ __forceinline__ float3x3(
        float m00, float m01, float m02,
        float m10, float m11, float m12, 
        float m20, float m21, float m22
    ) {
        m[0] = m00; m[1] = m01; m[2] = m02;
        m[3] = m10; m[4] = m11; m[5] = m12;
        m[6] = m20; m[7] = m21; m[8] = m22;
    }
    
    __device__ __forceinline__ float3x3(const float* data) {
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            m[i] = data[i];
        }
    }
};

// Matrix-vector operations
__device__ __forceinline__ float3 operator*(const float3x3& matrix, const float3& vec) {
    return make_float3(
        matrix.m[0] * vec.x + matrix.m[1] * vec.y + matrix.m[2] * vec.z,
        matrix.m[3] * vec.x + matrix.m[4] * vec.y + matrix.m[5] * vec.z,
        matrix.m[6] * vec.x + matrix.m[7] * vec.y + matrix.m[8] * vec.z
    );
}

// Legacy function for compatibility
__device__ __forceinline__ float3 mat3x3_mul_vec3(const float* matrix, const float3& vec) {
    return float3x3(matrix) * vec;
}

// float3 utility functions
__device__ __forceinline__ float3 float3_pow(const float3& v, float exp) {
    return make_float3(powf(v.x, exp), powf(v.y, exp), powf(v.z, exp));
}

__device__ __forceinline__ float3 float3_exp(const float3& v) {
    return make_float3(expf(v.x), expf(v.y), expf(v.z));
}

__device__ __forceinline__ float3 float3_log(const float3& v) {
    return make_float3(logf(v.x), logf(v.y), logf(v.z));
}

__device__ __forceinline__ float3 float3_max(const float3& v, float val) {
    return make_float3(fmaxf(v.x, val), fmaxf(v.y, val), fmaxf(v.z, val));
}

__device__ __forceinline__ float3 float3_min(const float3& v, float val) {
    return make_float3(fminf(v.x, val), fminf(v.y, val), fminf(v.z, val));
}

__device__ __forceinline__ float3 float3_add(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 float3_sub(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 float3_mul(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 float3_div(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__ float3 float3_mul_scalar(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ float3 float3_add_scalar(const float3& v, float s) {
    return make_float3(v.x + s, v.y + s, v.z + s);
}

__device__ __forceinline__ float3 float3_sub_scalar(const float3& v, float s) {
    return make_float3(v.x - s, v.y - s, v.z - s);
}

// float3 operators
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__ float3 operator+(const float3& v, float s) {
    return make_float3(v.x + s, v.y + s, v.z + s);
}

__device__ __forceinline__ float3 operator+(float s, const float3& v) {
    return make_float3(s + v.x, s + v.y, s + v.z);
}

__device__ __forceinline__ float3 operator-(const float3& v, float s) {
    return make_float3(v.x - s, v.y - s, v.z - s);
}

__device__ __forceinline__ float3 operator-(float s, const float3& v) {
    return make_float3(s - v.x, s - v.y, s - v.z);
}

__device__ __forceinline__ float3 operator*(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ __forceinline__ float3 operator/(const float3& v, float s) {
    return make_float3(v.x / s, v.y / s, v.z / s);
}

__device__ __forceinline__ float3 operator/(float s, const float3& v) {
    return make_float3(s / v.x, s / v.y, s / v.z);
}

// Convert float to clamped uint8
__device__ __forceinline__ uint8_t float_to_uint8(float x) {
    return (uint8_t)fminf(roundf(x * 255.0f), 255.0f);
}

// Helper functions with cleaner names
__device__ __forceinline__ float3 pow(const float3& v, float exp) {
    return make_float3(powf(v.x, exp), powf(v.y, exp), powf(v.z, exp));
}

__device__ __forceinline__ float3 exp(const float3& v) {
    return make_float3(expf(v.x), expf(v.y), expf(v.z));
}

__device__ __forceinline__ float3 fmax(const float3& v, float val) {
    return make_float3(fmaxf(v.x, val), fmaxf(v.y, val), fmaxf(v.z, val));
}

__device__ __forceinline__ float3 fmin(const float3& v, float val) {
    return make_float3(fminf(v.x, val), fminf(v.y, val), fminf(v.z, val));
}

// Clamp functions
__device__ __forceinline__ float3 clamp(const float3& v, float lo, float hi) {
    return make_float3(
        fminf(fmaxf(v.x, lo), hi),
        fminf(fmaxf(v.y, lo), hi), 
        fminf(fmaxf(v.z, lo), hi)
    );
}

// Normalized clamp to [0,1] - very common for color values
__device__ __forceinline__ float3 clamp01(const float3& v) {
    return clamp(v, 0.0f, 1.0f);
}

// Load/store float3 from/to arrays
__device__ __forceinline__ float3 float3_load(const float* input, int idx) {
    return make_float3(
        input[idx * 3 + 0],
        input[idx * 3 + 1], 
        input[idx * 3 + 2]
    );
}

__device__ __forceinline__ void float3_store(const float3& rgb, float* output, int idx) {
    output[idx * 3 + 0] = rgb.x;
    output[idx * 3 + 1] = rgb.y;
    output[idx * 3 + 2] = rgb.z;
}

// Convert float3 to uint8 RGB
__device__ __forceinline__ void float3_to_uint8_rgb(const float3& rgb, uint8_t* output, int idx) {
    output[idx * 3 + 0] = float_to_uint8(rgb.x);
    output[idx * 3 + 1] = float_to_uint8(rgb.y);
    output[idx * 3 + 2] = float_to_uint8(rgb.z);
}

// Linear interpolation
__device__ __forceinline__ float lerp(float t, float a, float b) {
    return a + t * (b - a);
}

__device__ __forceinline__ float3 lerp(float t, float3 a, float3 b) {
    return make_float3(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z));
}

// int3 utility functions
__device__ __forceinline__ int3 min(const int3& a, const int3& b) {
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

__device__ __forceinline__ float3 operator-(const float3& a, const int3& b) {
    return make_float3(a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z);
}

// Type conversion operators
__device__ __forceinline__ int3 make_int3(const float3& f) {
    return make_int3((int)f.x, (int)f.y, (int)f.z);
}

__device__ __forceinline__ float3 make_float3(const int3& i) {
    return make_float3((float)i.x, (float)i.y, (float)i.z);
}

// Flatten 3D index (x, y, z) for grid sized (sizex, sizey, sizez)
__device__ __forceinline__ int grid_index_2d(int x, int y, int z, int sizex, int sizey, int sizez) {
    return x + sizex * (y + sizey * z);
}

// Backward-compatible name used in some files
__device__ __forceinline__ int grid_index(int x, int y, int z, int sizex, int sizey, int sizez) {
    return grid_index_2d(x, y, z, sizex, sizey, sizez);
}

// RGB to grayscale conversion (Rec. 709)
__device__ __forceinline__ float rgb_to_gray(float3 rgb) {
    return rgb.x * 0.299f + rgb.y * 0.587f + rgb.z * 0.114f;
}


__device__ __forceinline__ void swap_floats(float& a, float& b) {
  const float tmp = b;
  b = a;
  a = tmp;
}

// Vectorized comparison operators 
struct bool3 {
    bool x, y, z;
    __device__ __forceinline__ bool3(bool x_, bool y_, bool z_) : x(x_), y(y_), z(z_) {}
};

// Comparison operators for float3
__device__ __forceinline__ bool3 operator>(const float3& a, float b) {
    return bool3(a.x > b, a.y > b, a.z > b);
}

__device__ __forceinline__ bool3 operator<(const float3& a, float b) {
    return bool3(a.x < b, a.y < b, a.z < b);
}

__device__ __forceinline__ bool3 operator>=(const float3& a, float b) {
    return bool3(a.x >= b, a.y >= b, a.z >= b);
}

__device__ __forceinline__ bool3 operator<=(const float3& a, float b) {
    return bool3(a.x <= b, a.y <= b, a.z <= b);
}

// Vectorized conditional selection: select(condition, true_val, false_val)
__device__ __forceinline__ float3 select(const bool3& cond, const float3& true_val, const float3& false_val) {
    return make_float3(
        cond.x ? true_val.x : false_val.x,
        cond.y ? true_val.y : false_val.y,
        cond.z ? true_val.z : false_val.z
    );
}
