#pragma once

#include "vector3D.cuh"
#include "point3D.cuh"

class matrix4D {

public:
	__host__ __device__ matrix4D(void) {
		m[0][0] = 1.0; m[0][1] = 0.0; m[0][2] = 0.0; m[0][3] = 0.0;
		m[1][0] = 0.0; m[1][1] = 1.0; m[1][2] = 0.0; m[1][3] = 0.0;
		m[2][0] = 0.0; m[2][1] = 0.0; m[2][2] = 1.0; m[2][3] = 0.0;
		m[3][0] = 0.0; m[3][1] = 0.0; m[3][2] = 0.0; m[3][3] = 1.0;
	}
	__host__ __device__ matrix4D(const matrix4D& mat);

	__host__ __device__ matrix4D::~matrix4D(void) {}

	__host__ __device__ float& operator ()(int i, int j) { return (m[i][j]); }
	__host__ __device__ const float& operator ()(int i, int j) const { return (m[i][j]); }

	__host__ __device__ matrix4D& operator =(const matrix4D& rhs);

	__host__ __device__ matrix4D operator *(const matrix4D& mat) const;
	__host__ __device__ matrix4D operator /(const float d);

	float	m[4][4];
};

__host__ __device__ matrix4D::matrix4D(const matrix4D& mat) {
	for (int x = 0; x < 4; x++) {
		for (int y = 0; y < 4; y++) {
			m[x][y] = mat.m[x][y];
		}
	}
}

__host__ __device__ matrix4D transponse(const matrix4D& mat) {
	matrix4D transp;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			transp.m[i][j] = mat.m[j][i];
			transp.m[j][i] = mat.m[i][j];
		}
	}
	return transp;
}

__host__ __device__ matrix4D& matrix4D::operator =(const matrix4D& rhs) {
	if (this == &rhs)
		return (*this);

	for (int x = 0; x < 4; x++)
		for (int y = 0; y < 4; y++)
			m[x][y] = rhs.m[x][y];

	return *this;
}

__host__ __device__ matrix4D matrix4D::operator *(const matrix4D& mat) const {
	matrix4D 	product;

	for (int y = 0; y < 4; y++)
		for (int x = 0; x < 4; x++) {
			float sum = 0.0;

			for (int j = 0; j < 4; j++)
				sum += m[x][j] * mat.m[j][y];

			product.m[x][y] = sum;
		}

	return (product);
}

__host__ __device__ matrix4D matrix4D::operator /(const float d) {
	for (int x = 0; x < 4; x++)
		for (int y = 0; y < 4; y++)
			m[x][y] = m[x][y] / d;

	return (*this);
}

__host__ __device__ void set_identity(matrix4D& mat) {
	mat.m[0][0] = 1.0; mat.m[0][1] = 0.0; mat.m[0][2] = 0.0; mat.m[0][3] = 0.0;
	mat.m[1][0] = 0.0; mat.m[1][1] = 1.0; mat.m[1][2] = 0.0; mat.m[1][3] = 0.0;
	mat.m[2][0] = 0.0; mat.m[2][1] = 0.0; mat.m[2][2] = 1.0; mat.m[2][3] = 0.0;
	mat.m[3][0] = 0.0; mat.m[3][1] = 0.0; mat.m[3][2] = 0.0; mat.m[3][3] = 1.0;
}


__host__ __device__ vector3D operator *(const matrix4D& mat, const vector3D& u) {
	return (vector3D(
		mat.m[0][0] * u[0] + mat.m[0][1] * u[1] + mat.m[0][2] * u[2],
		mat.m[1][0] * u[0] + mat.m[1][1] * u[1] + mat.m[1][2] * u[2],
		mat.m[2][0] * u[0] + mat.m[2][1] * u[1] + mat.m[2][2] * u[2])
	);
}

__host__ __device__ point3D operator *(const matrix4D& mat, const point3D& a) {
	return (point3D(
		mat.m[0][0] * a[0] + mat.m[0][1] * a[1] + mat.m[0][2] * a[2] + mat.m[0][3],
		mat.m[1][0] * a[0] + mat.m[1][1] * a[1] + mat.m[1][2] * a[2] + mat.m[1][3],
		mat.m[2][0] * a[0] + mat.m[2][1] * a[1] + mat.m[2][2] * a[2] + mat.m[2][3])
	);
}