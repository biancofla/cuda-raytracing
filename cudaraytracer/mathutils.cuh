#pragma once

#include "vector3D.cuh"
#include "const.cuh"

// CUDA dependencies.
#include "cuda_runtime.h"
#include "curand_kernel.h"

__device__ vector3D random_in_unit_sphere(curandState* local_rand_state) {
	vector3D p;
	do {
		p = 2.0f * RND_VEC - vector3D(1.0f, 1.0f, 1.0f);
	} while (dot(p, p) >= 1.0f);
	return p;
}

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vector3D& v, const vector3D& n, float ni_over_nt, vector3D& refracted) {
    vector3D uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
    }
	return false;
}

__device__ float radical_inverse_vdc(uint32_t n) {
	/**
	* Get the Van der Corput radical inverse.
	* 
	* The first line of the function, which reverses the bits of a 32
	* bit integer, swaps the lower 16 bits with the upper 16 bits of 
	* the value. The next line simultaneously swaps the first 8 bits 
	* of the result with the second 8 bits and the third 8 bits with 
	* the fourth. And so on, until the swapping of adjacent bits.
	*
	* Note: the way these bits are reversed is taken from the book
	* 'Hacker's Delight' of Henry S. Warren.
	* 
	* @param[n] number to invert.
	* @return radical inverse of n.
	*/
	n = (n << 16) | (n >> 16);
	n = ((n & 0x00FF00FFu) << 8) | ((n & 0xFF00FF00u) >> 8);
	n = ((n & 0x0F0F0F0Fu) << 4) | ((n & 0xF0F0F0F0u) >> 4);
	n = ((n & 0x33333333u) << 2) | ((n & 0xCCCCCCCCu) >> 2);
	n = ((n & 0x55555555u) << 1) | ((n & 0xAAAAAAAAu) >> 1);
	return float(n) * 2.3283064365386963e-10;
}

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }

__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }