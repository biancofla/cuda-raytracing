#pragma once

#include "point3D.cuh"
#include "color.cuh"

class simple_texture {
public:
	__device__ virtual color value(float u, float v, const point3D& p) const = 0;
};

class constant_texture : public simple_texture {
public:
	constant_texture() = default;
	__device__ constant_texture(color c) : color(c) {}
	
	__device__ virtual color value(float u, float v, const point3D& p) const {
		return color;
	}

	color color;
};

class checker_texture : public simple_texture {
public:
	checker_texture() = default;
	__device__ checker_texture(simple_texture* t0, simple_texture* t1) : even(t0), odd(t1) { }

	__device__ virtual color value(float u, float v, const point3D& p) const {
		float sines = sinf(10 * p[0]) * sinf(10 * p[1]) * sinf(10 * p[2]);
		if (sines < 0)
			return odd->value(u, v, p);
		else
			return even->value(u, v, p);
	}

	simple_texture* odd;
	simple_texture* even;
};