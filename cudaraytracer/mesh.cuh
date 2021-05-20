#pragma once

#include <iostream>
#include <vector>
#include <cstdio>

#include "obj_loader.h"
#include "vector3D.cuh"
#include "point3D.cuh"
#include "object.cuh"
#include "const.cuh"
#include "aabb.cuh"

using namespace std;

class mesh : public object {
public:
	__device__ mesh(point3D* v, vector3D* n, indx_struct* i, int nv, int nn, int ns, int nf, int ni) {
		vertices = v;
		normals = n;
		indices = i;

		num_vertices = nv;
		num_normals = nn;
		num_indices = ni;
		num_shapes = ns;
		num_faces = nf;

		aabb_mesh = aabb(
			point3D(FLT_MAX, FLT_MAX, FLT_MAX),
			point3D(FLT_MIN, FLT_MIN, FLT_MIN)
		);

		for (int v = 0; v < num_vertices; v++) {
			point3D p = vertices[v];

			aabb_mesh.pmin[0] = ffmin(p[0], aabb_mesh.pmin[0]);
			aabb_mesh.pmin[1] = ffmin(p[1], aabb_mesh.pmin[1]);
			aabb_mesh.pmin[2] = ffmin(p[2], aabb_mesh.pmin[2]);
			aabb_mesh.pmax[0] = ffmax(p[0], aabb_mesh.pmax[0]);
			aabb_mesh.pmax[1] = ffmax(p[1], aabb_mesh.pmax[1]);
			aabb_mesh.pmax[2] = ffmax(p[2], aabb_mesh.pmax[2]);
		}
	};

	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ virtual bool bounding_box(aabb& output_box) const;

	point3D* vertices;
	vector3D* normals;
	indx_struct* indices;

	int num_vertices;
	int num_normals;
	int num_indices;
	int num_shapes;
	int num_faces;

	aabb aabb_mesh;
};


__device__ bool triangle_intersection(const ray& r, float tmin, float tmax, hit_record& rec, const point3D& v0, const point3D& v1, const point3D& v2) {
	vector3D edge1, edge2, h, s, q;
	float a, f, u, v;

	edge1 = to_vector3D(v1 - v0);
	edge2 = to_vector3D(v2 - v0);

	h = cross(r.d, edge2);

	a = dot(edge1, h);
	if (a > -EPSILON && a < EPSILON) {
		return false;
	}

	f = 1.0f / a;
	s = to_vector3D(r.o - v0);
	u = f * (dot(s, h));
	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	q = cross(s, edge1);
	v = f * dot(r.d, q);
	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}

	float t = f * dot(edge2, q);
	if (t > tmin && t < tmax) {
		if (t > EPSILON) {
			rec.normal = -normalize(cross(to_vector3D(v1 - v0), to_vector3D(v2 - v0)));
			rec.t = dot(to_vector3D(v0 - r.o), rec.normal) / dot(r.direction(), rec.normal);
			rec.p = r.point_at_parameter(rec.t);
			return true;
		}
	}
	return false;
}


__device__ bool mesh::hit(const ray& ray, float t_min, float t_max, hit_record& rec) const {
	bool hit_anything = false;
	hit_record temp_rec;
	float closest_so_far = t_max;

	if (aabb_mesh.hit(ray, t_min, t_max) == false) {
		return false;
	}

	for (int v = 0; v < num_faces; v++) {
		int i0 = indices[3 * v + 0].vertex_indx;
		int i1 = indices[3 * v + 1].vertex_indx;
		int i2 = indices[3 * v + 2].vertex_indx;

		point3D v0 = vertices[i0];
		point3D v1 = vertices[i1];
		point3D v2 = vertices[i2];

		if (triangle_intersection(ray, t_min, closest_so_far, temp_rec, v0, v1, v2)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

__device__ bool mesh::bounding_box(aabb& output_box) const {
	output_box = aabb_mesh;
	return true;
}