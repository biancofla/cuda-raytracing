#pragma once

#include "mathutils.cuh"
#include "vector3D.cuh"
#include "object.cuh"

class box : public object {
public:
	box() = default;
	__device__ box(const point3D& p0, const point3D& p1) { box_min = p0; box_max = p1; }

	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

	__device__ virtual bool bounding_box(aabb& output_box) const;

	point3D box_min;
	point3D box_max;
};

__device__ bool box::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	for (int i = 0; i < 3; i++) {
		float t0 = ffmin(
			(box_min[i] - r.origin()[i]) / r.direction()[i],
			(box_max[i] - r.origin()[i]) / r.direction()[i]
		);
		float t1 = ffmax(
			(box_min[i] - r.origin()[i]) / r.direction()[i],
			(box_max[i] - r.origin()[i]) / r.direction()[i]
		);
		tmin = ffmax(t0, tmin);
		tmax = ffmin(t1, tmax);
		if (tmax <= tmin) return false;
	}
	return true;
}

__device__ bool box::bounding_box(aabb& output_box) const {
	output_box = aabb(box_min, box_max);
	return true;
}