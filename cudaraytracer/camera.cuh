#pragma once

#include "const.cuh"
#include "ray.cuh"

class camera {
public:
    __device__ camera(point3D _lookfrom, point3D _lookat, vector3D _up, float vfov, float aspect) {
        theta = vfov * M_PI / 180.0f;
        half_height = tan(theta / 2.0f);
        half_width = aspect * half_height;

        lookfrom = _lookfrom;
        lookat = _lookat;
        up = _up;

        w = normalize(to_vector3D(lookfrom - lookat));
        u = normalize(cross(up, w));
        v = cross(w, u);

        lower_left_corner = to_vector3D(lookfrom) - half_width * u - half_height * v - w;
        horizontal = 2.0f * half_width * u;
        vertical = 2.0f * half_height * v;
    }

    __device__ void update(point3D _lookfrom) {
        lookfrom = _lookfrom;

        w = normalize(to_vector3D(lookfrom - lookat));
        u = normalize(cross(up, w));
        v = cross(w, u);

        lower_left_corner = to_vector3D(lookfrom) - half_width * u - half_height * v - w;
        horizontal = 2.0f * half_width * u;
        vertical = 2.0f * half_height * v;
    }

    __device__ ray get_ray(float _u, float _v) { 
        return ray(lookfrom, lower_left_corner + _u * horizontal + _v * vertical - to_vector3D(lookfrom)); 
    }

    point3D lookfrom;
    point3D lookat;

    vector3D u, v, w;
    vector3D up;

    vector3D lower_left_corner;
    vector3D horizontal;
    vector3D vertical;

    float vfov, aspect;
    float theta, half_height, half_width;
};