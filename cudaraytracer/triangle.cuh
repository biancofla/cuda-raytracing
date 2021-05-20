#pragma once

#include "point3D.cuh"
#include "object.cuh"
#include "const.cuh"

class triangle : public object {
public:
	__device__ triangle() {
		v0 = point3D( 0.0f, 1.0f, 0.0f);
		v1 = point3D(-1.0f, 0.0f, 0.0f);
		v2 = point3D( 1.0f, 0.0f, 0.0f);
	}
	__device__ triangle(point3D _vertices[3]) {
		v0 = _vertices[0];
		v1 = _vertices[1];
		v2 = _vertices[2];
		norm = normalize(cross(to_vector3D(v1 - v0), to_vector3D(v2 - v0)));
	}
	__device__ triangle(point3D _v0, point3D _v1, point3D _v2) {
		v0 = _v0;
		v1 = _v1;
		v2 = _v2;
		norm = normalize(cross(to_vector3D(v1 - v0), to_vector3D(v2 - v0)));
	};

	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

	__device__ virtual bool bounding_box(aabb& output_box) const;

	point3D v0, v1, v2;
	vector3D norm;
};

__device__ bool triangle::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	vector3D edge1, edge2, h, s, q;
	float a, f, u, v;

	edge1 = to_vector3D(v1 - v0);
	edge2 = to_vector3D(v2 - v0);

	h = cross(r.d, edge2);

	a = dot(edge1, h);
	if (a > -EPSILON && a < EPSILON) return false;

	f = 1.0f / a;
	s = to_vector3D(r.o - v0);

	u = f * (dot(s, h));
	if (u < 0.0f || u > 1.0f) return false;

	q = cross(s, edge1);

	v = f * dot(r.d, q);
	if (v < 0.0f || u + v > 1.0f) return false;

	float t = f * dot(edge2, q);
	if (t > tmin && t < tmax) {
		if (t > EPSILON) {
			point3D intersPoint = r.o + to_point3D((normalize(r.d) * (t * magnitude(r.d))));
			rec.normal = norm;
			rec.t = dot(to_vector3D(v0 - r.o), norm) / dot(r.direction(), norm);
			rec.p = intersPoint;
			return true;
		}
	}
	return false;
}

__device__ bool triangle::bounding_box(aabb& output_box) const {
	output_box.pmax = point3D(FLT_MIN, FLT_MIN, FLT_MIN);
	output_box.pmin = point3D(FLT_MAX, FLT_MAX, FLT_MAX);

	output_box.pmin[0] = ffmin(v0[0], output_box.pmin[0]);
	output_box.pmin[1] = ffmin(v0[1], output_box.pmin[1]);
	output_box.pmin[2] = ffmin(v0[2], output_box.pmin[2]);
	output_box.pmin[0] = ffmin(v1[0], output_box.pmin[0]);
	output_box.pmin[1] = ffmin(v1[1], output_box.pmin[1]);
	output_box.pmin[2] = ffmin(v1[2], output_box.pmin[2]);
	output_box.pmin[0] = ffmin(v2[0], output_box.pmin[0]);
	output_box.pmin[1] = ffmin(v2[1], output_box.pmin[1]);
	output_box.pmin[2] = ffmin(v2[2], output_box.pmin[2]);

	output_box.pmax[0] = ffmax(v0[0], output_box.pmax[0]);
	output_box.pmax[1] = ffmax(v0[1], output_box.pmax[1]);
	output_box.pmax[2] = ffmax(v0[2], output_box.pmax[2]);
	output_box.pmax[0] = ffmax(v1[0], output_box.pmax[0]);
	output_box.pmax[1] = ffmax(v1[1], output_box.pmax[1]);
	output_box.pmax[2] = ffmax(v1[2], output_box.pmax[2]);
	output_box.pmax[0] = ffmax(v2[0], output_box.pmax[0]);
	output_box.pmax[1] = ffmax(v2[1], output_box.pmax[1]);
	output_box.pmax[2] = ffmax(v2[2], output_box.pmax[2]);

	return true;
}