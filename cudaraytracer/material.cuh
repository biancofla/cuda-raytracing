#pragma once

#include "mathutils.cuh"
#include "simple_texture.cuh"
#include "object.cuh"
#include "color.cuh"
#include "ray.cuh"

struct hit_record;

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const {
        return false;
    }

    __device__ virtual color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3D& p) const {
        return color(0.0f, 0.0f, 0.0f);
    }

};

class lambertian : public material {
public:
    __device__ lambertian(simple_texture *a) : albedo(a) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const {
        vector3D target = to_vector3D(rec.p) + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - to_vector3D(rec.p));
        attenuation = albedo->value(0.0f, 0.0f, rec.p);

        return true;
    }

    simple_texture* albedo;
};

class metal : public material {
public:
    __device__ metal(const color& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const {
        vector3D reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;

        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

    color albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const {
        vector3D outward_normal;
        vector3D reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = color(1.0, 1.0, 1.0);
        vector3D refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / magnitude(r_in.direction());
            cosine = sqrtf(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / magnitude(r_in.direction());
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
            reflect_prob = schlick(cosine, ref_idx);
        }
        else {
            reflect_prob = 1.0f;
        }
        if (curand_uniform(local_rand_state) < reflect_prob) {
            scattered = ray(rec.p, reflected);
        } 
        else {
            scattered = ray(rec.p, refracted);
        }
        return true;
    }

    float ref_idx;
};

class diffuse_light : public material {
public:
    __device__ diffuse_light(simple_texture* a) : emit(a) {}

    __device__ virtual color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3D& p) const {
        return emit->value(u, v, p);
    }

    simple_texture* emit;
};
