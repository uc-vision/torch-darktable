#pragma once

#include <cuda_runtime.h>
#include "device_math.h"

// sRGB gamma correction
__device__ __forceinline__ float srgb_to_linear(float x) {
    return (x > 0.04045f) ? powf((x + 0.055f) / 1.055f, 2.4f) : x / 12.92f;
}

__device__ __forceinline__ float3 srgb_to_linear(float3 rgb) {
    const float threshold = 0.04045f;
    const float a = 0.055f;
    const float gamma = 2.4f;
    const float linear_scale = 1.0f / 12.92f;
    
    bool3 use_pow = rgb > threshold;
    float3 pow_result = pow((rgb + a) / (1.0f + a), gamma);
    float3 linear_result = rgb * linear_scale;
    
    return select(use_pow, pow_result, linear_result);
}

__device__ __forceinline__ float3 linear_to_srgb(float3 linear_rgb) {
    const float threshold = 0.0031308f;
    const float a = 0.055f;
    const float gamma = 1.0f / 2.4f;
    const float srgb_scale = 12.92f;
    
    bool3 use_pow = linear_rgb > threshold;
    float3 pow_result = (1.0f + a) * pow(linear_rgb, gamma) - a;
    float3 linear_result = linear_rgb * srgb_scale;
    
    return select(use_pow, pow_result, linear_result);
}

// LAB f functions
__device__ __forceinline__ float lab_f(float t) {
    return (t > 0.008856f) ? powf(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
}

__device__ __forceinline__ float3 lab_f(float3 t) {
    const float threshold = 0.008856f;
    const float linear_mult = 7.787f;
    const float linear_add = 16.0f/116.0f;
    
    bool3 use_pow = t > threshold;
    float3 pow_result = pow(t, 1.0f/3.0f);
    float3 linear_result = t * linear_mult + linear_add;
    
    return select(use_pow, pow_result, linear_result);
}

__device__ __forceinline__ float lab_f_inv(float t) {
    float t_cubed = t * t * t;
    return (t_cubed > 0.008856f) ? t_cubed : (t - 16.0f/116.0f) / 7.787f;
}

__device__ __forceinline__ float3 lab_f_inv(float3 t) {
    float3 t_cubed = t * t * t;
    const float threshold = 0.008856f;
    const float linear_coeff = 16.0f/116.0f;
    
    bool3 use_cube = t_cubed > threshold;
    float3 cube_result = t_cubed;
    float3 linear_result = (t - linear_coeff) / 7.787f;
    
    return select(use_cube, cube_result, linear_result);
}


// Color space conversions
__device__ float3 rgb_to_xyz(float3 rgb) {
    float3 linear_rgb = srgb_to_linear(rgb);
    
    const float3x3 xyz_matrix(
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f
    );
    
    return xyz_matrix * linear_rgb;
}

__device__ float3 xyz_to_lab(float3 xyz) {
    const float3 d65_white = make_float3(0.95047f, 1.0f, 1.08883f);
    xyz = xyz / d65_white;
    
    float3 f_xyz = lab_f(xyz);
    
    // Use pre-scaled constants for normalized LAB output
    float L = (116.0f/100.0f) * f_xyz.y - (16.0f/100.0f);     // L: 0-1
    float a = (500.0f/128.0f) * (f_xyz.x - f_xyz.y);          // A: -1 to +1  
    float b = (200.0f/128.0f) * (f_xyz.y - f_xyz.z);          // B: -1 to +1
    
    return make_float3(L, a, b);
}

__device__ float3 lab_to_xyz(float3 lab) {
    // Use pre-scaled constants for normalized LAB input
    float fy = lab.x * (100.0f/116.0f) + (16.0f/116.0f);
    float3 f_xyz = make_float3(
        lab.y * (128.0f/500.0f) + fy,
        fy,
        fy - lab.z * (128.0f/200.0f)
    );
    
    float3 xyz = lab_f_inv(f_xyz);
    
    const float3 d65_white = make_float3(0.95047f, 1.0f, 1.08883f);
    return xyz * d65_white;
}

__device__ float3 xyz_to_rgb(float3 xyz) {
    const float3x3 xyz_to_rgb_matrix(
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f
    );
    
    float3 linear_rgb = xyz_to_rgb_matrix * xyz;
    return linear_to_srgb(linear_rgb);
}

__device__ float3 xyz_to_linear_rgb(float3 xyz) {
    const float3x3 xyz_to_rgb_matrix(
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f
    );
    
    return xyz_to_rgb_matrix * xyz;  // Stay in linear space
}

__device__ float3 rgb_to_lab(float3 rgb) {
    return xyz_to_lab(rgb_to_xyz(rgb));
}

__device__ float3 lab_to_rgb(float3 lab) {
    return xyz_to_rgb(lab_to_xyz(lab));
}

// HSL color space conversions
__device__ float3 rgb_to_hsl(float3 rgb) {
    float max_val = fmaxf(fmaxf(rgb.x, rgb.y), rgb.z);
    float min_val = fminf(fminf(rgb.x, rgb.y), rgb.z);
    float delta = max_val - min_val;
    
    float h = 0.0f;
    float s = 0.0f;
    float l = (max_val + min_val) * 0.5f;
    
    if (delta > 1e-6f) {
        s = (l < 0.5f) ? delta / (max_val + min_val) : delta / (2.0f - max_val - min_val);
        
        if (max_val == rgb.x) {
            h = (rgb.y - rgb.z) / delta + (rgb.y < rgb.z ? 6.0f : 0.0f);
        } else if (max_val == rgb.y) {
            h = (rgb.z - rgb.x) / delta + 2.0f;
        } else {
            h = (rgb.x - rgb.y) / delta + 4.0f;
        }
        h /= 6.0f;
    }
    
    return make_float3(h, s, l);
}

__device__ float hsl_hue_to_rgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f/6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f/2.0f) return q;
    if (t < 2.0f/3.0f) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
    return p;
}

__device__ float3 hsl_to_rgb(float3 hsl) {
    float h = hsl.x;
    float s = hsl.y;
    float l = hsl.z;
    
    if (s < 1e-6f) {
        return make_float3(l, l, l);
    }
    
    float q = (l < 0.5f) ? l * (1.0f + s) : l + s - l * s;
    float p = 2.0f * l - q;
    
    float r = hsl_hue_to_rgb(p, q, h + 1.0f/3.0f);
    float g = hsl_hue_to_rgb(p, q, h);
    float b = hsl_hue_to_rgb(p, q, h - 1.0f/3.0f);
    
    return make_float3(r, g, b);
}

__device__ float rgb_to_lab_l(float3 rgb) {
    float3 linear_rgb = srgb_to_linear(rgb);
    
    // Compute Y component directly (luminance)
    float y_xyz = 0.2126729f * linear_rgb.x + 0.7151522f * linear_rgb.y + 0.0721750f * linear_rgb.z;
    
    float fy = lab_f(y_xyz);
    float L = (116.0f/100.0f) * fy - (16.0f/100.0f);
    
    return fmaxf(0.0f, L);
}

__device__ float3 color_transform_3x3(float3 color, const float3x3& matrix) {
    return clipf(matrix * color);
}

__device__ float3 modify_rgb_luminance(float3 rgb, float luminance) {
    float3 lab = rgb_to_lab(rgb);
    float clamped = fmaxf(0.0f, fminf(1.0f, luminance));
    float3 new_lab = make_float3(clamped, lab.y, lab.z);  // L is now 0-1 range
    return clipf(lab_to_rgb(new_lab));
}

__device__ float3 modify_rgb_log_luminance(float3 rgb, float log_luminance, float eps) {
    float3 lab = rgb_to_lab(rgb);
    float lum = fmaxf(0.0f, fminf(1.0f, expf(log_luminance)));
    float3 new_lab = make_float3(lum, lab.y, lab.z);  // L is now 0-1 range
    return clipf(lab_to_rgb(new_lab));
}

__device__ float3 modify_rgb_hsl(float3 rgb, float hue_adjust = 0.0f, float sat_adjust = 0.0f, float lum_adjust = 0.0f) {
    float3 hsl = rgb_to_hsl(rgb);
    
    float new_h = hsl.x + hue_adjust;
    if (new_h < 0.0f) new_h += 1.0f;
    if (new_h > 1.0f) new_h -= 1.0f;
    
    float new_s = pow(hsl.y, 1.0f / (1.0f + sat_adjust));
    float new_l = pow(hsl.z, 1.0f / (1.0f + lum_adjust));
    
    float3 new_hsl = make_float3(new_h, new_s, new_l);
    return clipf(hsl_to_rgb(new_hsl));
}

// Darktable-style vibrance - perceptually superior to HSL saturation
__device__ float3 modify_rgb_vibrance_dt(float3 rgb, float amount = 0.0f) {
    float3 lab = rgb_to_lab(rgb);
    
    // Calculate chroma (colorfulness) in LAB space
    // With normalized LAB (A,B in -1 to +1), chroma is already in reasonable range
    const float chroma = sqrtf(lab.y * lab.y + lab.z * lab.z);
    
    // Darktable's vibrance algorithm:
    // - More colorful areas get more enhancement
    // - Lightness is slightly reduced as chroma increases (more natural)
    const float ls = 1.0f - amount * chroma * 0.25f;  // lightness scaling
    const float ss = 1.0f + amount * chroma;          // saturation scaling
    
    float new_l = lab.x * ls;
    float new_a = lab.y * ss;
    float new_b = lab.z * ss;
    
    float3 new_lab = make_float3(new_l, new_a, new_b);
    return clipf(lab_to_rgb(new_lab));
}
