#pragma once

#include "device_math.h"

// All color conversion device functions as __forceinline__ to avoid multiple definition errors

__device__ __forceinline__ float3 linear_to_srgb(float3 linear) {
    return make_float3(
        linear.x <= 0.0031308f ? 12.92f * linear.x : 1.055f * powf(linear.x, 1.0f / 2.4f) - 0.055f,
        linear.y <= 0.0031308f ? 12.92f * linear.y : 1.055f * powf(linear.y, 1.0f / 2.4f) - 0.055f,
        linear.z <= 0.0031308f ? 12.92f * linear.z : 1.055f * powf(linear.z, 1.0f / 2.4f) - 0.055f
    );
}

__device__ __forceinline__ float3 srgb_to_linear(float3 srgb) {
    return make_float3(
        srgb.x <= 0.04045f ? srgb.x / 12.92f : powf((srgb.x + 0.055f) / 1.055f, 2.4f),
        srgb.y <= 0.04045f ? srgb.y / 12.92f : powf((srgb.y + 0.055f) / 1.055f, 2.4f),
        srgb.z <= 0.04045f ? srgb.z / 12.92f : powf((srgb.z + 0.055f) / 1.055f, 2.4f)
    );
}

__device__ __forceinline__ float3 rgb_to_xyz(float3 rgb) {
    float3 linear_rgb = srgb_to_linear(rgb);
    
    const float3x3 rgb_to_xyz_matrix(
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f
    );
    
    return rgb_to_xyz_matrix * linear_rgb;
}

__device__ __forceinline__ float lab_f(float t) {
    const float delta = 6.0f / 29.0f;
    const float delta_cubed = delta * delta * delta;
    const float factor = 1.0f / (3.0f * delta * delta);
    const float offset = 4.0f / 29.0f;
    
    return (t > delta_cubed) ? cbrtf(t) : factor * t + offset;
}

__device__ __forceinline__ float lab_f_inv(float t) {
    const float delta = 6.0f / 29.0f;
    const float factor = 3.0f * delta * delta;
    const float offset = 4.0f / 29.0f;
    
    return (t > delta) ? (t * t * t) : factor * (t - offset);
}

__device__ __forceinline__ float3 xyz_to_lab(float3 xyz) {
    const float3 d65_white = make_float3(0.95047f, 1.0f, 1.08883f);
    float3 normalized = xyz / d65_white;
    
    float fx = lab_f(normalized.x);
    float fy = lab_f(normalized.y);
    float fz = lab_f(normalized.z);
    
    float L = 116.0f * fy - 16.0f;
    float a = 500.0f * (fx - fy);
    float b = 200.0f * (fy - fz);
    
    return make_float3(L / 100.0f, a / 128.0f, b / 128.0f);
}

__device__ __forceinline__ float3 lab_to_xyz(float3 lab) {
    float L = lab.x * 100.0f;
    float a = lab.y * 128.0f;
    float b = lab.z * 128.0f;
    
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b / 200.0f;
    
    float3 xyz = make_float3(
        lab_f_inv(fx),
        lab_f_inv(fy),
        lab_f_inv(fz)
    );
    
    const float3 d65_white = make_float3(0.95047f, 1.0f, 1.08883f);
    return xyz * d65_white;
}

__device__ __forceinline__ float3 xyz_to_rgb(float3 xyz) {
    const float3x3 xyz_to_rgb_matrix(
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f
    );
    
    float3 linear_rgb = xyz_to_rgb_matrix * xyz;
    return linear_to_srgb(linear_rgb);
}

__device__ __forceinline__ float3 xyz_to_linear_rgb(float3 xyz) {
    const float3x3 xyz_to_rgb_matrix(
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f
    );
    
    return xyz_to_rgb_matrix * xyz;
}

__device__ __forceinline__ float3 rgb_to_lab(float3 rgb) {
    return xyz_to_lab(rgb_to_xyz(rgb));
}

__device__ __forceinline__ float3 lab_to_rgb(float3 lab) {
    return xyz_to_rgb(lab_to_xyz(lab));
}

__device__ __forceinline__ float3 rgb_to_hsl(float3 rgb) {
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

__device__ __forceinline__ float hsl_hue_to_rgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f/6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f/2.0f) return q;
    if (t < 2.0f/3.0f) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
    return p;
}

__device__ __forceinline__ float3 hsl_to_rgb(float3 hsl) {
    float h = hsl.x;
    float s = hsl.y;
    float l = hsl.z;
    
    if (s == 0.0f) {
        return make_float3(l, l, l);
    }
    
    float q = (l < 0.5f) ? l * (1.0f + s) : l + s - l * s;
    float p = 2.0f * l - q;
    
    return make_float3(
        hsl_hue_to_rgb(p, q, h + 1.0f/3.0f),
        hsl_hue_to_rgb(p, q, h),
        hsl_hue_to_rgb(p, q, h - 1.0f/3.0f)
    );
}

__device__ __forceinline__ float rgb_to_lab_l(float3 rgb) {
    return rgb_to_lab(rgb).x;
}

__device__ __forceinline__ float3 color_transform_3x3(float3 color, const float3x3& matrix) {
    return matrix * color;
}

__device__ __forceinline__ float3 modify_rgb_luminance(float3 rgb, float new_luminance) {
    float3 lab = rgb_to_lab(rgb);
    lab.x = new_luminance;
    return clip(lab_to_rgb(lab));
}

__device__ __forceinline__ float3 modify_rgb_log_luminance(float3 rgb, float log_luminance, float eps) {
    float3 lab = rgb_to_lab(rgb);
    lab.x = expf(log_luminance + eps);
    return clip(lab_to_rgb(lab));
}

__device__ __forceinline__ float3 modify_rgb_hsl(float3 rgb, float hue_adjust, float sat_adjust, float lum_adjust) {
    float3 hsl = rgb_to_hsl(rgb);
    float3 new_hsl = make_float3(
        fmodf(hsl.x + hue_adjust + 1.0f, 1.0f),
        fmaxf(0.0f, fminf(1.0f, hsl.y + sat_adjust)),
        fmaxf(0.0f, fminf(1.0f, hsl.z + lum_adjust))
    );
    return clip(hsl_to_rgb(new_hsl));
}

// Darktable-style vibrance - perceptually superior to HSL saturation
__device__ __forceinline__ float3 modify_rgb_vibrance_dt(float3 rgb, float amount = 0.0f) {
    float3 lab = rgb_to_lab(rgb);
    
    // Calculate chroma (colorfulness) in LAB space
    const float chroma = sqrtf(lab.y * lab.y + lab.z * lab.z);
    
    // Darktable's vibrance algorithm:
    // - More colorful areas get more enhancement
    // - Lightness is slightly reduced as chroma increases (more natural)
    const float ls = 1.0f - amount * chroma * 0.25f;  // lightness scaling
    const float ss = 1.0f + amount * chroma;          // saturation scaling
    
    float3 new_lab = make_float3(lab.x * ls, lab.y * ss, lab.z * ss);
    return clip(lab_to_rgb(new_lab));
}
