#pragma once

#include "matrix4D.cuh"
#include "object.cuh"
#include "const.cuh"

class instance: public object {
public:
	instance() = default;
	__device__ instance(object * nobj_ptr, material * m);

	__device__ instance* instance::clone(void) const;

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

	__device__ virtual bool bounding_box(aabb& output_box) const;

	__device__ void identity();

	__device__ void translate(const vector3D& trans);
	__device__ void translate(const float dx, const float dy, const float dz);

	__device__ void scale(const vector3D& s);
	__device__ void scale(const float a, const float b, const float c);

	__device__ void rotate_x(const float r);
	__device__ void rotate_y(const float r);
	__device__ void rotate_z(const float r);

	__device__ void setMaterial(material* m) { mat = m; }
	__device__ material* getMaterial() { return(mat); }

	object* object_ptr;
	matrix4D inverse_matrix;
	matrix4D current_matrix;
	material* mat;
};

__device__ instance* instance::clone(void) const {
	return (new instance(*this));
}

__device__ instance::instance(object* nobj_ptr, material* m) {
	object_ptr = nobj_ptr;
	mat = m;
}

__device__ bool instance::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	ray inv_ray;

	inv_ray.o = inverse_matrix * r.o;
	inv_ray.d = inverse_matrix * r.d;

	if (object_ptr->hit(inv_ray, t_min, t_max, rec)) {
		rec.p = current_matrix * inv_ray.point_at_parameter(rec.t);
		rec.normal = normalize(transponse(inverse_matrix) * rec.normal);
		rec.mat = mat;

		return true;
	}
	return false;
}

__device__ bool instance::bounding_box(aabb& output_box) const {
	object_ptr->bounding_box(output_box);
	output_box.pmin = current_matrix * output_box.pmin;
	output_box.pmax = current_matrix * output_box.pmax;
	return true;
}

__device__ void instance::identity() {
	set_identity(current_matrix);
	set_identity(inverse_matrix);
}

__device__ void instance::translate(const vector3D& trans) {
	translate(trans[0], trans[1], trans[2]);
}

__device__ void instance::translate(const float dx, const float dy, const float dz) {
	matrix4D inv_translation_matrix;

	inv_translation_matrix.m[0][3] = -dx;
	inv_translation_matrix.m[1][3] = -dy;
	inv_translation_matrix.m[2][3] = -dz;

	inverse_matrix = inverse_matrix * inv_translation_matrix;

	matrix4D translation_matrix;

	translation_matrix.m[0][3] = dx;
	translation_matrix.m[1][3] = dy;
	translation_matrix.m[2][3] = dz;

	current_matrix = translation_matrix * current_matrix;
}

__device__ void instance::scale(const vector3D& s) {
	scale(s[0], s[1], s[2]);
}

__device__ void instance::scale(const float a, const float b, const float c) {
	matrix4D inv_scaling_matrix;

	inv_scaling_matrix.m[0][0] = 1.0f / a;
	inv_scaling_matrix.m[1][1] = 1.0f / b;
	inv_scaling_matrix.m[2][2] = 1.0f / c;

	inverse_matrix = inverse_matrix * inv_scaling_matrix;

	matrix4D scaling_matrix;

	scaling_matrix.m[0][0] = a;
	scaling_matrix.m[1][1] = b;
	scaling_matrix.m[2][2] = c;

	current_matrix = scaling_matrix * current_matrix;
}

__device__ void instance::rotate_x(const float theta) {
	float sin_theta = sin(theta * DEG2RAD);
	float cos_theta = cos(theta * DEG2RAD);

	matrix4D inv_x_rotation_matrix;

	inv_x_rotation_matrix.m[1][1] = cos_theta;
	inv_x_rotation_matrix.m[1][2] = sin_theta;
	inv_x_rotation_matrix.m[2][1] = -sin_theta;
	inv_x_rotation_matrix.m[2][2] = cos_theta;

	inverse_matrix = inverse_matrix * inv_x_rotation_matrix;

	matrix4D x_rotation_matrix;

	x_rotation_matrix.m[1][1] = cos_theta;
	x_rotation_matrix.m[1][2] = -sin_theta;
	x_rotation_matrix.m[2][1] = sin_theta;
	x_rotation_matrix.m[2][2] = cos_theta;

	current_matrix = x_rotation_matrix * current_matrix;
}

__device__ void instance::rotate_y(const float theta) {
	float sin_theta = sin(theta * DEG2RAD);
	float cos_theta = cos(theta * DEG2RAD);

	matrix4D inv_y_rotation_matrix;

	inv_y_rotation_matrix.m[0][0] = cos_theta;
	inv_y_rotation_matrix.m[0][2] = -sin_theta;
	inv_y_rotation_matrix.m[2][0] = sin_theta;
	inv_y_rotation_matrix.m[2][2] = cos_theta;

	inverse_matrix = inverse_matrix * inv_y_rotation_matrix;

	matrix4D y_rotation_matrix;

	y_rotation_matrix.m[0][0] = cos_theta;
	y_rotation_matrix.m[0][2] = sin_theta;
	y_rotation_matrix.m[2][0] = -sin_theta;
	y_rotation_matrix.m[2][2] = cos_theta;

	current_matrix = current_matrix * y_rotation_matrix;
}

__device__ void instance::rotate_z(const float theta) {
	float sin_theta = sin(theta * DEG2RAD);
	float cos_theta = cos(theta * DEG2RAD);

	matrix4D inv_z_rotation_matrix;

	inv_z_rotation_matrix.m[0][0] = cos_theta;
	inv_z_rotation_matrix.m[0][1] = sin_theta;
	inv_z_rotation_matrix.m[1][0] = -sin_theta;
	inv_z_rotation_matrix.m[1][1] = cos_theta;

	inverse_matrix = inverse_matrix * inv_z_rotation_matrix;

	matrix4D z_rotation_matrix;

	z_rotation_matrix.m[0][0] = cos_theta;
	z_rotation_matrix.m[0][1] = -sin_theta;
	z_rotation_matrix.m[1][0] = sin_theta;
	z_rotation_matrix.m[1][1] = cos_theta;

	current_matrix = z_rotation_matrix * current_matrix;
}