#pragma once

#include <iostream>
#include <vector>
#include <string>

#include "vector3D.cuh"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace tinyobj;

struct indx_struct {
	int vertex_indx;
	int normal_indx;
	int texcoord_indx;
};

bool obj_lookup(const char* file_name, const char* base_path, int& num_vertices, int& num_normals, int& num_shapes, int& num_faces, int& num_indices) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, file_name, base_path, true);

	if (!err.empty()) {
		printf("5s\n", err);
		return false;
	}

	if (!ret) {
		printf("Failed to load/parse .obj.\n");
		return false;
	}

	num_vertices = attrib.vertices.size() / 3;

	num_normals = attrib.normals.size() / 3;

	num_shapes = shapes.size();

	num_faces = 0;

	num_indices = 0;
	// For each shape.
	for (int s = 0; s < num_shapes; s++) {
		// For each face.
		for (int f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			num_faces++;
			int fv = shapes[s].mesh.num_face_vertices[f];
			// For each vertex in the face.
			for (int v = 0; v < fv; v++) {
				num_indices++;
			}
		}
	}

	return true;
}

bool load_mesh(const char* file_name, const char* base_path, point3D* &vertices, vector3D* &normals, indx_struct* &indices, int &num_vertices, int &num_normals, int &num_shapes, int &num_faces, int &num_indices) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, file_name, base_path, true);

	if (!err.empty()) {
		printf("5s\n", err);
		return false;
	}

	if (!ret) {
		printf("Failed to load/parse .obj.\n");
		return false;
	}

	num_vertices = attrib.vertices.size() / 3;
	vertices = new point3D[num_vertices];

	num_normals = attrib.normals.size() / 3;
	normals = new vector3D[num_normals];

	num_shapes = shapes.size();

	num_faces = 0;

	num_indices = 0;

	for (int v = 0; v < num_vertices; v++) {
		vertices[v] = point3D(
			attrib.vertices[3 * v + 0],
			attrib.vertices[3 * v + 1],
			attrib.vertices[3 * v + 2]
		);
	}

	for (int v = 0; v < num_normals; v++) {
		normals[v] = vector3D(
			attrib.normals[3 * v + 0],
			attrib.normals[3 * v + 1],
			attrib.normals[3 * v + 2]
		);
	}

	for (int s = 0; s < num_shapes; s++) {
		for (int f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			num_faces++;

			int fv = shapes[s].mesh.num_face_vertices[f];
			for (int v = 0; v < fv; v++) {
				num_indices++;
			}
		}
	}

	indices = new indx_struct[num_indices];

	int i = 0;
	// For each shape.
	for (int s = 0; s < num_shapes; s++) {
		int index_offset = 0;
		// For each face.
		for (int f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];
			// For each vertex in the face.
			for (int v = 0; v < fv; v++) {
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				indices[i] = { idx.vertex_index, idx.normal_index, idx.texcoord_index };

				i++;
			}
			index_offset += fv;
		}
	}

	return true;
}