#pragma once

#include "triangle.cuh"
#include "object.cuh"

class rectangle : public object {
public:
	rectangle() = default;
	__device__ rectangle(point3D v0, point3D v1, point3D v2, point3D v3) : tri_a(v0, v1, v3), tri_b(v1, v2, v3) {};

	__device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;

	__device__ virtual bool bounding_box(aabb& output_box) const;

	triangle tri_a, tri_b;
};

__device__ bool rectangle::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	if (tri_a.hit(r, tmin, tmax, rec))
		return true;
	return tri_b.hit(r, tmin, tmax, rec);
}

__device__ bool rectangle::bounding_box(aabb& output_box) const {
	aabb box_a, box_b;

	tri_a.bounding_box(box_a);
	tri_b.bounding_box(box_b);

	output_box = surrounding_box(box_a, box_b);

	return true;
}