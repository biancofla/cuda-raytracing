#pragma once

// CUDA dependencies.
#include "cuda_runtime.h"

class point3D {
public:
	point3D() = default;
	__host__ __device__ point3D(float x, float y, float z) { v[0] = x; v[1] = y; v[2] = z; }

	__host__ __device__ inline float x() const { return v[0]; }
	__host__ __device__ inline float y() const { return v[1]; }
	__host__ __device__ inline float z() const { return v[2]; }

	__host__ __device__ inline float  operator [](int i) const { return v[i]; }
	__host__ __device__ inline float& operator [](int i) { return v[i]; };

	float v[3];
};

__host__ __device__ inline point3D operator +(const point3D& a, const point3D& b)
{
	return point3D(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

__host__ __device__ inline point3D operator -(const point3D& a, const point3D& b)
{
	return point3D(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

__host__ __device__ inline point3D operator *(const float s, const point3D& a)
{
	return point3D(a[0] * s, a[1] * s, a[2] * s);
}