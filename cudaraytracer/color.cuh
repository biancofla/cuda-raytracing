#pragma once

#include <math.h>

// CUDA dependencies.
#include "cuda_runtime.h"

class color {
public:
    color() = default;
    __host__ __device__ color(float r, float g, float b) { v[0] = r; v[1] = g; v[2] = b; }

    __host__ __device__ inline float r() const { return v[0]; }
    __host__ __device__ inline float g() const { return v[1]; }
    __host__ __device__ inline float b() const { return v[2]; }

    __host__ __device__ inline const color& operator +() const { return *this; }
    __host__ __device__ inline color operator -() const { return color(-v[0], -v[1], -v[2]); }
    __host__ __device__ inline float  operator [](int i) const { return v[i]; }
    __host__ __device__ inline float& operator [](int i) { return v[i]; };

    __host__ __device__ inline color& operator +=(const color& u);
    __host__ __device__ inline color& operator -=(const color& u);
    __host__ __device__ inline color& operator *=(float s);
    __host__ __device__ inline color& operator /=(float s);

    float v[3];
};

__host__ __device__ color& color::operator +=(const color& u)
{
    v[0] += u.v[0];
    v[1] += u.v[1];
    v[2] += u.v[2];
    return *this;
}

__host__ __device__ color& color::operator -=(const color& u)
{
    v[0] -= u.v[0];
    v[1] -= u.v[1];
    v[2] -= u.v[2];
    return *this;
}

__host__ __device__ color& color::operator *=(float s)
{
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
    return *this;
}

__host__ __device__ color& color::operator /=(float s)
{
    s = 1.0f / s;
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
    return *this;
}

__host__ __device__ inline color operator +(const color& a, const color& b)
{
    return color(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

__host__ __device__ inline color operator -(const color& a, const color& b)
{
    return color(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

__host__ __device__ inline color operator *(const color& a, const color& b)
{
    return color(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

__host__ __device__ inline color operator *(const color& u, float s)
{
    return color(u[0] * s, u[1] * s, u[2] * s);
}

__host__ __device__ inline color operator *(float s, const color& u)
{
    return color(u[0] * s, u[1] * s, u[2] * s);
}

__host__ __device__ inline color operator /(const color& u, float s)
{
    s = 1.0f / s;
    return color(u[0] * s, u[1] * s, u[2] * s);
}

__host__ __device__ inline float magnitude(const color& u)
{
    return sqrtf(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

__host__ __device__ inline color normalize(const color& u)
{
    return u / magnitude(u);
}

__host__ __device__ inline color clamp(color& c) {
    c.v[0] = (c.v[0] > 1.0f) ? 1.0f : ((c.v[0] < 0.0f) ? 0.0f : c.v[0]);
    c.v[1] = (c.v[1] > 1.0f) ? 1.0f : ((c.v[1] < 0.0f) ? 0.0f : c.v[1]);
    c.v[2] = (c.v[2] > 1.0f) ? 1.0f : ((c.v[2] < 0.0f) ? 0.0f : c.v[2]);
    return c;
}
