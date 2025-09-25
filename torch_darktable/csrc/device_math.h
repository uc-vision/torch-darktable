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

__device__ __host__ inline int3 operator-(const int3& a, int scalar) {
    return make_int3(a.x - scalar, a.y - scalar, a.z - scalar);
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





// Helper functions
__device__ __forceinline__ float fsquare(float x) {
    return x * x;
}

__device__ __forceinline__ float sqrf(float x) {
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
    
    __host__ __device__ __forceinline__ float3x3(
        float m00, float m01, float m02,
        float m10, float m11, float m12, 
        float m20, float m21, float m22
    ) {
        m[0] = m00; m[1] = m01; m[2] = m02;
        m[3] = m10; m[4] = m11; m[5] = m12;
        m[6] = m20; m[7] = m21; m[8] = m22;
    }
    
    __host__ __device__ __forceinline__ float3x3(const float* data) {
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


// Mathematical function overloads for vector types

// Template-based mathematical functions
template<typename F>
__device__ __forceinline__ float2 apply_unary(const float2& v, F func) {
    return make_float2(func(v.x), func(v.y));
}

template<typename F>
__device__ __forceinline__ float3 apply_unary(const float3& v, F func) {
    return make_float3(func(v.x), func(v.y), func(v.z));
}

template<typename F>
__device__ __forceinline__ float4 apply_unary(const float4& v, F func) {
    return make_float4(func(v.x), func(v.y), func(v.z), func(v.w));
}

// Mathematical function overloads - back to simple implementations
__device__ __forceinline__ float2 pow(const float2& v, float exp) {
    return make_float2(powf(v.x, exp), powf(v.y, exp));
}

__device__ __forceinline__ float3 pow(const float3& v, float exp) {
    return make_float3(powf(v.x, exp), powf(v.y, exp), powf(v.z, exp));
}

__device__ __forceinline__ float4 pow(const float4& v, float exp) {
    return make_float4(powf(v.x, exp), powf(v.y, exp), powf(v.z, exp), powf(v.w, exp));
}

__device__ __forceinline__ float2 exp(const float2& v) {
    return make_float2(expf(v.x), expf(v.y));
}

__device__ __forceinline__ float3 exp(const float3& v) {
    return make_float3(expf(v.x), expf(v.y), expf(v.z));
}

__device__ __forceinline__ float4 exp(const float4& v) {
    return make_float4(expf(v.x), expf(v.y), expf(v.z), expf(v.w));
}

__device__ __forceinline__ float2 log(const float2& v) {
    return make_float2(logf(v.x), logf(v.y));
}

__device__ __forceinline__ float3 log(const float3& v) {
    return make_float3(logf(v.x), logf(v.y), logf(v.z));
}

__device__ __forceinline__ float4 log(const float4& v) {
    return make_float4(logf(v.x), logf(v.y), logf(v.z), logf(v.w));
}

// Template for binary operations with scalar
template<typename F>
__device__ __forceinline__ float2 apply_binary_scalar(const float2& v, float val, F func) {
    return make_float2(func(v.x, val), func(v.y, val));
}

template<typename F>
__device__ __forceinline__ float3 apply_binary_scalar(const float3& v, float val, F func) {
    return make_float3(func(v.x, val), func(v.y, val), func(v.z, val));
}

template<typename F>
__device__ __forceinline__ float4 apply_binary_scalar(const float4& v, float val, F func) {
    return make_float4(func(v.x, val), func(v.y, val), func(v.z, val), func(v.w, val));
}

template<typename F>
__device__ __forceinline__ int2 apply_binary_scalar(const int2& v, int val, F func) {
    return make_int2(func(v.x, val), func(v.y, val));
}

template<typename F>
__device__ __forceinline__ int3 apply_binary_scalar(const int3& v, int val, F func) {
    return make_int3(func(v.x, val), func(v.y, val), func(v.z, val));
}

// max/min overloads - simple implementations
__device__ __forceinline__ float2 max(const float2& v, float val) {
    return make_float2(fmaxf(v.x, val), fmaxf(v.y, val));
}

__device__ __forceinline__ float3 max(const float3& v, float val) {
    return make_float3(fmaxf(v.x, val), fmaxf(v.y, val), fmaxf(v.z, val));
}

__device__ __forceinline__ float4 max(const float4& v, float val) {
    return make_float4(fmaxf(v.x, val), fmaxf(v.y, val), fmaxf(v.z, val), fmaxf(v.w, val));
}

__device__ __forceinline__ int2 max(const int2& v, int val) {
    return make_int2(max(v.x, val), max(v.y, val));
}

__device__ __forceinline__ int3 max(const int3& v, int val) {
    return make_int3(max(v.x, val), max(v.y, val), max(v.z, val));
}

__device__ __forceinline__ float2 min(const float2& v, float val) {
    return make_float2(fminf(v.x, val), fminf(v.y, val));
}

__device__ __forceinline__ float3 min(const float3& v, float val) {
    return make_float3(fminf(v.x, val), fminf(v.y, val), fminf(v.z, val));
}

__device__ __forceinline__ float4 min(const float4& v, float val) {
    return make_float4(fminf(v.x, val), fminf(v.y, val), fminf(v.z, val), fminf(v.w, val));
}

__device__ __forceinline__ int2 min(const int2& v, int val) {
    return make_int2(min(v.x, val), min(v.y, val));
}

__device__ __forceinline__ int3 min(const int3& v, int val) {
    return make_int3(min(v.x, val), min(v.y, val), min(v.z, val));
}

__device__ __forceinline__ int2 min(const int2& a, const int2& b) {
    return make_int2(min(a.x, b.x), min(a.y, b.y));
}

__device__ __forceinline__ int3 min(const int3& a, const int3& b) {
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

__device__ __forceinline__ int2 max(const int2& a, const int2& b) {
    return make_int2(max(a.x, b.x), max(a.y, b.y));
}

__device__ __forceinline__ int3 max(const int3& a, const int3& b) {
    return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

// clamp overloads
__host__ __device__ __forceinline__ float clamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__host__ __device__ __forceinline__ float2 clamp(const float2& v, float lo, float hi) {
    return make_float2(clamp(v.x, lo, hi), clamp(v.y, lo, hi));
}

__host__ __device__ __forceinline__ float3 clamp(const float3& v, float lo, float hi) {
    return make_float3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}

__host__ __device__ __forceinline__ float4 clamp(const float4& v, float lo, float hi) {
    return make_float4(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi), clamp(v.w, lo, hi));
}


__host__ __device__ __forceinline__ int2 clamp(const int2& v, int lo, int hi) {
    return make_int2(clamp(v.x, lo, hi), clamp(v.y, lo, hi));
}

__host__ __device__ __forceinline__ int3 clamp(const int3& v, int lo, int hi) {
    return make_int3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}

// clip overloads (0.0 to 1.0)
__device__ __forceinline__ float clip(float v) {
    return clamp(v, 0.0f, 1.0f);
}

__device__ __forceinline__ float2 clip(const float2& v) {
    return clamp(v, 0.0f, 1.0f);
}

__device__ __forceinline__ float3 clip(const float3& v) {
    return clamp(v, 0.0f, 1.0f);
}

__device__ __forceinline__ float4 clip(const float4& v) {
    return clamp(v, 0.0f, 1.0f);
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


// Load/store overloads for vector types

// Generic load function that infers type from context
template<typename T>
__device__ __forceinline__ T load(const float* input, int idx);

template<>
__device__ __forceinline__ float2 load<float2>(const float* input, int idx) {
    return make_float2(input[idx * 2 + 0], input[idx * 2 + 1]);
}

template<>
__device__ __forceinline__ float3 load<float3>(const float* input, int idx) {
    return make_float3(input[idx * 3 + 0], input[idx * 3 + 1], input[idx * 3 + 2]);
}

template<>
__device__ __forceinline__ float4 load<float4>(const float* input, int idx) {
    return make_float4(input[idx * 4 + 0], input[idx * 4 + 1], input[idx * 4 + 2], input[idx * 4 + 3]);
}

// store overloads
__device__ __forceinline__ void store(const float2& v, float* output, int idx) {
    output[idx * 2 + 0] = v.x;
    output[idx * 2 + 1] = v.y;
}

__device__ __forceinline__ void store(const float3& v, float* output, int idx) {
    output[idx * 3 + 0] = v.x;
    output[idx * 3 + 1] = v.y;
    output[idx * 3 + 2] = v.z;
}

__device__ __forceinline__ void store(const float4& v, float* output, int idx) {
    output[idx * 4 + 0] = v.x;
    output[idx * 4 + 1] = v.y;
    output[idx * 4 + 2] = v.z;
    output[idx * 4 + 3] = v.w;
}

// store as uint8 overloads
__device__ __forceinline__ void store(const float3& v, uint8_t* output, int idx) {
    output[idx * 3 + 0] = float_to_uint8(v.x);
    output[idx * 3 + 1] = float_to_uint8(v.y);
    output[idx * 3 + 2] = float_to_uint8(v.z);
}

__device__ __forceinline__ void store(const float4& v, uint8_t* output, int idx) {
    output[idx * 4 + 0] = float_to_uint8(v.x);
    output[idx * 4 + 1] = float_to_uint8(v.y);
    output[idx * 4 + 2] = float_to_uint8(v.z);
    output[idx * 4 + 3] = float_to_uint8(v.w);
}

// Linear interpolation
__device__ __forceinline__ float lerp(float t, float a, float b) {
    return a + t * (b - a);
}

__device__ __forceinline__ float3 lerp(float t, float3 a, float3 b) {
    return make_float3(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z));
}


__device__ __forceinline__ float3 operator-(const float3& a, const int3& b) {
    return make_float3(a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z);
}

// Type conversion functions
__device__ __forceinline__ int2 to_int(const float2& f) {
    return make_int2((int)f.x, (int)f.y);
}

__device__ __forceinline__ int3 to_int(const float3& f) {
    return make_int3((int)f.x, (int)f.y, (int)f.z);
}

__device__ __forceinline__ int4 to_int(const float4& f) {
    return make_int4((int)f.x, (int)f.y, (int)f.z, (int)f.w);
}

__device__ __forceinline__ float2 to_float(const int2& i) {
    return make_float2((float)i.x, (float)i.y);
}

__device__ __forceinline__ float3 to_float(const int3& i) {
    return make_float3((float)i.x, (float)i.y, (float)i.z);
}

__device__ __forceinline__ float4 to_float(const int4& i) {
    return make_float4((float)i.x, (float)i.y, (float)i.z, (float)i.w);
}

// Flatten 3D index (x, y, z) for grid sized (size.x, size.y, size.z)
__device__ __forceinline__ int grid_index(int x, int y, int z, int3 size) {
    return x + size.x * (y + size.y * z);
}

__device__ __forceinline__ int grid_index(int3 pos, int3 size) {
    return pos.x + size.x * (pos.y + size.y * pos.z);
}

// Backward-compatible versions (deprecated)
__device__ __forceinline__ int grid_index(int x, int y, int z, int sizex, int sizey, int sizez) {
    return grid_index(x, y, z, make_int3(sizex, sizey, sizez));
}

// RGB to grayscale conversion (Rec. 709)
__device__ __forceinline__ float rgb_to_gray(float3 rgb) {
    return rgb.x * 0.299f + rgb.y * 0.587f + rgb.z * 0.114f;
}


template<typename T>
__device__ __forceinline__ void swap(T& a, T& b) {
  const T tmp = b;
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

// float4 utility functions and operators
struct bool4 {
    bool x, y, z, w;
    __device__ __forceinline__ bool4(bool x_, bool y_, bool z_, bool w_) : x(x_), y(y_), z(z_), w(w_) {}
};

__device__ __forceinline__ bool4 operator<(const float4& a, float b) {
    return bool4(a.x < b, a.y < b, a.z < b, a.w < b);
}

__device__ __forceinline__ float4 select(const bool4& cond, const float4& true_val, const float4& false_val) {
    return make_float4(
        cond.x ? true_val.x : false_val.x,
        cond.y ? true_val.y : false_val.y,
        cond.z ? true_val.z : false_val.z,
        cond.w ? true_val.w : false_val.w
    );
}

__device__ __forceinline__ float4 abs(const float4& v) {
    return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

// float4 arithmetic operators
__device__ __forceinline__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ __forceinline__ float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ __forceinline__ float4 operator/(const float4& a, const float4& b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__device__ __forceinline__ float4 operator+(const float4& v, float s) {
    return make_float4(v.x + s, v.y + s, v.z + s, v.w + s);
}

__device__ __forceinline__ float4 operator-(const float4& v, float s) {
    return make_float4(v.x - s, v.y - s, v.z - s, v.w - s);
}

__device__ __forceinline__ float4 operator*(const float4& v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

__device__ __forceinline__ float4 operator*(float s, const float4& v) {
    return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}

__device__ __forceinline__ float4 operator-(float s, const float4& v) {
    return make_float4(s - v.x, s - v.y, s - v.z, s - v.w);
}

// Assignment operators for float3
__device__ __forceinline__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __forceinline__ float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__device__ __forceinline__ float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

__device__ __forceinline__ float3& operator/=(float3& a, const float3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

__device__ __forceinline__ float3& operator+=(float3& a, float s) {
    a.x += s;
    a.y += s;
    a.z += s;
    return a;
}

__device__ __forceinline__ float3& operator*=(float3& a, float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}
