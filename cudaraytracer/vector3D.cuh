#pragma once

#include <math.h>

#include "point3D.cuh"

// CUDA dependencies.
#include "cuda_runtime.h"

class vector3D {
public:
    vector3D() = default;
    __host__ __device__ vector3D(float x, float y, float z) { v[0] = x; v[1] = y; v[2] = z; }

    __host__ __device__ inline float x() const { return v[0]; }
    __host__ __device__ inline float y() const { return v[1]; }
    __host__ __device__ inline float z() const { return v[2]; }

    __host__ __device__ inline const vector3D& operator +() const { return *this; }
    __host__ __device__ inline vector3D operator -() const { return vector3D(-v[0], -v[1], -v[2]); }
    __host__ __device__ inline float  operator [](int i) const { return v[i]; }
    __host__ __device__ inline float& operator [](int i) { return v[i]; };

    __host__ __device__ inline vector3D& operator +=(const vector3D& u);
    __host__ __device__ inline vector3D& operator -=(const vector3D& u);
    __host__ __device__ inline vector3D& operator *=(float s);
    __host__ __device__ inline vector3D& operator /=(float s);

    float v[3];
};

__host__ __device__ vector3D& vector3D::operator +=(const vector3D& u)
{
    v[0] += u.v[0];
    v[1] += u.v[1];
    v[2] += u.v[2];
    return *this;
}

__host__ __device__ vector3D& vector3D::operator -=(const vector3D& u)
{
    v[0] -= u.v[0];
    v[1] -= u.v[1];
    v[2] -= u.v[2];
    return *this;
}

__host__ __device__ vector3D& vector3D::operator *=(float s)
{
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
    return *this;
}

__host__ __device__ vector3D& vector3D::operator /=(float s)
{
    s = 1.0f / s;
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
    return *this;
}

__host__ __device__ inline vector3D operator +(const vector3D& u, const vector3D& w)
{
    return vector3D(u[0] + w[0], u[1] + w[1], u[2] + w[2]);
}

__host__ __device__ inline vector3D operator -(const vector3D& u, const vector3D& w)
{
    return vector3D(u[0] - w[0], u[1] - w[1], u[2] - w[2]);
}

__host__ __device__ inline vector3D operator *(const vector3D& u, float s)
{
    return vector3D(u[0] * s, u[1] * s, u[2] * s);
}

__host__ __device__ inline vector3D operator *(float s, const vector3D& u)
{
    return vector3D(u[0] * s, u[1] * s, u[2] * s);
}

__host__ __device__ inline vector3D operator /(const vector3D& u, float s)
{
    s = 1.0f / s;
    return vector3D(u[0] * s, u[1] * s, u[2] * s);
}

__host__ __device__ inline float magnitude(const vector3D& u)
{
    return sqrtf(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

__host__ __device__ inline vector3D normalize(const vector3D& u)
{
    return u / magnitude(u);
}

__host__ __device__ inline float dot(const vector3D& a, const vector3D& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__ inline vector3D cross(const vector3D& u, const vector3D& w)
{
    return (vector3D(
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0])
    );
}

__host__ __device__ inline vector3D project(const vector3D& u, const vector3D& w)
{
    return w * (dot(u, w) / dot(w, w));
}

__host__ __device__ inline vector3D reject(const vector3D& u, const vector3D& w)
{
    return u - w * (dot(u, w) / dot(w, w));
}

__host__ __device__ inline vector3D reflect(const vector3D& i, const vector3D& n)
{
    return  i - 2.0f * dot(i, n) * n;
}

__host__ __device__ inline vector3D to_vector3D(const point3D& a) {
    return vector3D(a[0], a[1], a[2]);
}

__host__ __device__ inline point3D to_point3D(const vector3D& u) {
    return point3D(u[0], u[1], u[2]);
}