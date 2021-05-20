#pragma once

#include "vector3D.cuh"
#include "point3D.cuh"
#include "object.cuh"

class sphere : public object {
public:
	__device__ sphere() {
		center = point3D(0.0f, 0.0f, 0.0f);
		radius = 1.0f;
	}
	__device__ sphere(point3D cen, float r, material* m) : center(cen), radius(r) {};

	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

	__device__ virtual bool bounding_box(aabb& output_box) const;
	
	point3D center;
	float radius;
};

__device__ bool sphere::hit(const ray& ray, float t_min, float t_max, hit_record& rec) const {
	vector3D oc = to_vector3D(ray.origin() - center);
	float a = dot(ray.direction(), ray.direction());
	float b = dot(oc, ray.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;

	if (discriminant < 0) return false;

	float sqrtd = sqrt(discriminant);

	float root = (-b - sqrtd) / a;
	if (root < t_min || t_max < root) {
		root = (-b + sqrtd) / a;
		if (root < t_min || t_max < root) return false;
	}

	rec.t = root;
	rec.p = ray.point_at_parameter(rec.t);
	rec.normal = to_vector3D(rec.p - center) / radius;

	return true;
}

__device__ bool sphere::bounding_box(aabb& output_box) const {
	output_box = aabb(
		center - point3D(radius, radius, radius),
		center + point3D(radius, radius, radius)
	);
	return true;
}