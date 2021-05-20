#pragma once

#include "vector3D.cuh"
#include "point3D.cuh"

// CUDA dependencies.
#include "cuda_runtime.h"

class ray {
public:
    ray() = default;
    __device__ ray(const point3D& _o, const vector3D& _d) { o = _o; d = _d; }

    __device__ point3D origin() const { return o; }
    __device__ vector3D direction() const { return d; }
    __device__ point3D point_at_parameter(float t) const { return o + t * to_point3D(d); }

    point3D o;
    vector3D d;
};