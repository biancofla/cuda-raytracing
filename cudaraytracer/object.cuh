#pragma once

#include "aabb.cuh"
#include "ray.cuh"

class material;

struct hit_record
{
    float t;
    float u;
    float v;
    point3D p;
    vector3D normal;
    material* mat;
};

class object {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;

    __device__ virtual bool bounding_box(aabb& output_box) const = 0;
};