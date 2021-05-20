#pragma once

#include "mathutils.cuh"
#include "ray.cuh"

#include "thrust/swap.h"

using namespace thrust;

class aabb {
public:
	aabb() = default;
	__device__ aabb(const point3D& a, const point3D& b) { pmin = a; pmax = b; }

	__device__ bool hit(const ray& r, float tmin, float tmax) const;

	point3D pmin;
	point3D pmax;
};

__device__ bool aabb::hit(const ray& r, float t_min, float t_max) const {
	for (int a = 0; a < 3; a++) {
		float inv_d = 1.0f / r.direction()[0];

		float t_0 = (pmin[0] - r.origin()[0]) * inv_d;
		float t_1 = (pmax[0] - r.origin()[0]) * inv_d;

		if (inv_d < 0.0f) {
			swap(t_0, t_1);
		}

		t_min = t_0 > t_min ? t_0 : t_min;
		t_max = t_1 < t_max ? t_1 : t_max;
		if (t_max <= t_min) {
			return false;
		}
	}
	return true;
}

__device__ aabb surrounding_box(aabb box_0, aabb box_1) {
	point3D small_point(
		ffmin(box_0.pmin[0], box_1.pmin[0]),
		ffmin(box_0.pmin[1], box_1.pmin[1]),
		ffmin(box_0.pmin[2], box_1.pmin[2])
	);
	point3D big_point(
		ffmax(box_0.pmax[0], box_1.pmax[0]),
		ffmax(box_0.pmax[1], box_1.pmax[1]),
		ffmax(box_0.pmax[2], box_1.pmax[2])
	);
	return aabb(small_point, big_point);
}