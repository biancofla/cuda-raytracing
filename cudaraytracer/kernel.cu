#include <iostream>
#include <time.h>

#include "object_list.cuh"
#include "mathutils.cuh"
#include "rectangle.cuh"
#include "material.cuh"
#include "triangle.cuh"
#include "instance.cuh"
#include "obj_loader.h"
#include "object.cuh"
#include "camera.cuh"
#include "sphere.cuh"
#include "raster.cuh"
#include "color.cuh"
#include "const.cuh"
#include "mesh.cuh"
#include "box.cuh"
#include "ray.cuh"
#include "bvh.cuh"

#include "SDL.h"

// CUDA dependencies.
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

using namespace std;

#define HANDLE_CUDA_ERROR(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	/**
	* Check CUDA error of a specified API call.
	*
	* @param[result] CUDA error type as a code.
	* @param[func] Name of the caller function.
	* @param[file] Name of the file from which the function is called.
	* @param[line] Line number from which the function is called.
	*/
    if (result) {
		printf(
			"[CUDA ERROR] Code %d, at line %d, of the file %s, in function %s.\n",
			static_cast<unsigned int>(result),
			file,
			line,
			func
		);
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color shot(const ray& r, object** world, curandState* local_rand_state) {
	/**
	* Computer pixel color using iterative path tracing algorithm.
	* 
	* @param[r] shot ray.
	* @param[world] pointer of the scene.
	* @param[local_rand_state] CUDA rand state.
	* @return color of the hitted pixel.
	*/
	ray cur_ray = r;
	color cur_attenuation(1.0f, 1.0f, 1.0f);
	color cur_emitted(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < RAY_BOUNCES; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			color attenuation;
			color emitted = rec.mat->emitted(r, rec, rec.u, rec.v, rec.p);
			if (rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation = cur_attenuation * attenuation;
				cur_emitted = cur_emitted + emitted * cur_attenuation;
				cur_ray = scattered;
			}
			else {
				return cur_emitted + emitted * cur_attenuation;
			}
		}
		else {
			return cur_emitted;
		}
	}
	return cur_emitted;
}

__global__ void rand_init(curandState* rand_state) {
	/**
	* Init CUDA rand state for each tread (this rand state will be 
	* used for world creation).
	* 
	* @param[rand_state] CUDA rand state.
	*/
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1996, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	/**
	* Init CUDA rand state for each tread.
	*
	* @param[max_x] Maximum thread size along x axis.
	* @param[max_y] Maximum thread size along y axis.
	* @param[rand_state] CUDA rand state.
	*/
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	// Each thread gets different seed and same sequence.
	curand_init(1996 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(color* frame_buffer, int max_x, int max_y, int ns, object** world, camera** cam, curandState* rand_state) {
	/**
	* Render out frame buffer.
	*
	* @param[frame_buffer] frame buffer that will contain pixels colors.
	* @param[max_x] Maximum thread size along x axis.
	* @param[max_y] Maximum thread size along y axis.
	* @param[world] pointer of the scene.
	* @param[cam] pointer of the cam.
	* @param[rand_state] CUDA rand state.
	*/
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i;

	curandState local_rand_state = rand_state[pixel_index];

	color col(0.0f, 0.0f, 0.0f);

	// Hammersley sampling.
	for (int s = 1; s < ns + 1; s++) {
		float x_i = 1.0f / s;

		float phi_i = radical_inverse_vdc(s);

		float u = float(i + x_i  ) / float(max_x);
		float v = float(j + phi_i) / float(max_y);

		ray r = (*cam)->get_ray(u, v);
		col += shot(r, world, &local_rand_state);
	}

	rand_state[pixel_index] = local_rand_state;

	col = col / float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);

	frame_buffer[pixel_index] = col;
}

__global__ void three_spheres(object** objects, object** world, camera** cam, curandState* rand_state) {
	/**
	* Init scene containing three spheres.
	* 
	* @param[objects] pointer list that will contain the objects.
	* @param[world] pointer of the scene.
	* @param[cam] pointer of the cam.
	* @param[rand_state] CUDA rand state.
	*/
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		object* sphere_model = new sphere();

		instance* sphere_ptr = new instance(
			sphere_model, 
			new diffuse_light(new constant_texture(color(2.0f, 2.0f, 2.0f)))
		);
		sphere_ptr->scale(0.5f, 0.5f, 0.5f);
		sphere_ptr->translate(0.0f, 0.0f, -1.0f);
		*(objects) = sphere_ptr;

		sphere_ptr = new instance(
			sphere_model,
			new lambertian(new constant_texture(color(0.8f, 0.8f, 0.0f)))
		);
		sphere_ptr->scale(100.0f, 100.0f, 100.0f);
		sphere_ptr->translate(0.0f, -100.5f, -1.0f);
		*(objects + 1) = sphere_ptr;

		sphere_ptr = new instance(
			sphere_model,
			new metal(color(0.8f, 0.6f, 0.2f), 0.0f)
		);
		sphere_ptr->scale(0.5f, 0.5f, 0.5f);
		sphere_ptr->translate(1.0f, 0.0f, -1.0f);
		*(objects + 2) = sphere_ptr;

		sphere_ptr = new instance(
			sphere_model,
			new dielectric(1.5f)
		);
		sphere_ptr->scale(0.5f, 0.5f, 0.5f);
		sphere_ptr->translate(-1.0f, 0.0f, -1.0f);
		*(objects + 3) = sphere_ptr;
		
		if (USE_BVH == 1) {
			*world = new bvh(objects, 0, 4, rand_state);
		}
		else {
			*world = new object_list(objects, 4);
		}

		*cam = new camera(
			point3D(-10.0f, 10.0f, 8.0f),
			point3D(0.0f, 0.0f, -1.0f),
			vector3D(0.0f, 1.0f, 0.0f),
			20.0f,
			float(RES_X) / float(RES_Y)
		);
	}
}

__global__ void random_spheres(object** objects, object** world, camera** cam, curandState* rand_state) {
	/**
	* Init scene containing three giant sphere plus many little spheres around them.
	* 
	* @param[objects] pointer list that will contain the objects.
	* @param[world] pointer of the scene.
	* @param[cam] pointer of the cam.
	* @param[rand_state] CUDA rand state.
	*/
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;

		object* sphere_model = new sphere();

		// Floor.
		instance* sphere_ptr = new instance(
			sphere_model,
			new lambertian(new checker_texture(new constant_texture(color(0.0f, 0.0f, 0.0f)), new constant_texture(color(1.0f, 1.0f, 1.0f))))
		);
		sphere_ptr->scale(1000.0f, 1000.0f, 1000.0f);
		sphere_ptr->translate(0.0f, -1000.0f, -1.0f);
		*(objects) = sphere_ptr;

		// Little spheres.
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = RND;

				if (choose_mat < 0.5f) {
					sphere_ptr = new instance(
						sphere_model,
						new diffuse_light(new constant_texture(color(RND, RND, RND)))
					);
				} else if (choose_mat < 0.8f) {
					sphere_ptr = new instance(
						sphere_model,
						new lambertian(new constant_texture(color(RND * RND, RND * RND, RND * RND)))
					);
				}
				else if (choose_mat < 0.95f) {
					sphere_ptr = new instance(
						sphere_model,
						new metal(color(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND)
					);
				}
				else {
					sphere_ptr = new instance(
						sphere_model,
						new dielectric(1.5)
					);
				}
				sphere_ptr->scale(0.2f, 0.2f, 0.2f);
				sphere_ptr->translate(a + RND, 0.2f, b + RND);
				*(objects + i) = sphere_ptr;

				i++;
			}
		}

		// Giant spheres.
		sphere_ptr = new instance(
			sphere_model,
			new dielectric(1.5)
		);
		sphere_ptr->translate(0.0f, 1.0f, 0.0f);
		*(objects + (i++)) = sphere_ptr;

		sphere_ptr = new instance(
			sphere_model,
			new lambertian(new constant_texture(color(0.4f, 0.2f, 0.1f)))
		);
		sphere_ptr->translate(-4.0f, 1.0f, 0.0f);
		*(objects + (i++)) = sphere_ptr;

		sphere_ptr = new instance(
			sphere_model,
			new metal(color(0.7f, 0.6f, 0.5f), 0.0f)
		);
		sphere_ptr->translate(4.0f, 1.0f, 0.0f);
		*(objects + (i++)) = sphere_ptr;

		if (USE_BVH == 1) {
			*world = new bvh(objects, 0, i, rand_state);
		}
		else {
			*world = new object_list(objects, i);
		}

		*rand_state = local_rand_state;

		*cam = new camera(
			point3D(13.0f, 2.0f, 3.0f),
			point3D(0.0f, 0.0f, 0.0f),
			vector3D(0.0f, 1.0f, 0.0f),
			30.0f,
			float(RES_X) / float(RES_Y)
		);
	}
}

__global__ void cornell_box(object** objects, object** world, camera** cam, curandState* rand_state) {
	/**
	* Init Cornell box scene.
	* 
	* @param[objects] pointer list that will contain the objects.
	* @param[world] pointer of the scene.
	* @param[cam] pointer of the cam.
	* @param[rand_state] CUDA rand state.
	*/
	material* red   = new lambertian(new constant_texture(color(0.65f, 0.05f, 0.05f)));
	material* white = new lambertian(new constant_texture(color(0.73f, 0.73f, 0.73f)));
	material* green = new lambertian(new constant_texture(color(0.12f, 0.45f, 0.15f)));
	material* light = new diffuse_light(new constant_texture(color(2.0f, 2.0f, 2.0f)));

	// Back wall.
	object* back_wall = new rectangle(
		point3D(549.6f, 0.0f, 559.2f),
		point3D(0.0f, 0.0f, 559.2f),
		point3D(0.0f, 548.8f, 559.2f),
		point3D(556.0f, 548.8f, 559.2f)
	);
	instance* wall_ptr = new instance(
		back_wall,
		white
	);
	*(objects) = wall_ptr;

	// Right wall.
	object* right_wall = new rectangle(
		point3D(0.0f, 0.0f, 559.2f),
		point3D(0.0f, 0.0f, 0.0f),
		point3D(0.0f, 548.8f, 0.0f),
		point3D(0.0f, 548.8f, 559.2f)
	);
	wall_ptr = new instance(
		right_wall,
		green
	);
	*(objects + 1) = wall_ptr;

	// Left wall.
	object* left_wall = new rectangle(
		point3D(552.8f, 0.0f, 0.0f),
		point3D(549.6f, 0.0f, 559.2f),
		point3D(556.0f, 548.8f, 559.2f),
		point3D(556.0f, 548.8f, 0.0f)
	);
	wall_ptr = new instance(
		left_wall,
		red
	);
	*(objects + 2) = wall_ptr;

	// Ceil.
	object* ceil = new rectangle(
		point3D(556.0f, 548.8f, 0.0f),
		point3D(556.0f, 548.8f, 559.2f),
		point3D(0.0f, 548.8f, 559.2f),
		point3D(0.0f, 548.8f, 0.0f)
	);
	wall_ptr = new instance(
		ceil,
		white
	);
	*(objects + 3) = wall_ptr;
	
	// Floor.
	object* floor = new rectangle(
		point3D(552.8f, 0.0f, 0.0f),
		point3D(0.0f, 0.0f, 0.0f),
		point3D(0.0f, 0.0f, 559.2f),
		point3D(549.6f, 0.0f, 559.2f)
	);
	wall_ptr = new instance(
		floor,
		white
	);
	*(objects + 4) = wall_ptr;

	// Box and sphere.
	object* sphere_model = new sphere();

	instance* sphere_ptr = new instance(
		sphere_model,
		new metal(color(0.7f, 0.6f, 0.5f), 0.0f)
	);
	sphere_ptr->scale(100.0f, 100.0f, 100.0f);
	sphere_ptr->translate(375.0f, 110.f, 275.0f);
	*(objects + 5) = sphere_ptr;

	sphere_ptr = new instance(
		sphere_model,
		new dielectric(1.5f)
	);
	sphere_ptr->scale(100.0f, 100.0f, 100.0f);
	sphere_ptr->translate(150.0f, 110.0f, 250.0f);
	*(objects + 6) = sphere_ptr;

	// Light.
	object* ceiling_light = new rectangle(
		point3D(343.0f, 548.0f, 227.0f), 
		point3D(343.0f, 548.0f, 332.0f), 
		point3D(213.0f, 548.0f, 332.0f), 
		point3D(213.0f, 548.0f, 227.0f)
	);
	*(objects + 7) = new instance(ceiling_light, light);

	if (USE_BVH == 1) {
		*world = new bvh(objects, 0, 8, rand_state);
	}
	else {
		*world = new object_list(objects, 8);
	}

	*cam = new camera(
		point3D(278.0f, 278.0f, -800.0f),
		point3D(278.0f, 278.0f, 0.0f),
		vector3D(0.0f, 1.0f, 0.0f),
		40.0f,
		float(RES_X) / float(RES_Y)
	);
}

__global__ void dragon(object** objects, object** world, camera** cam, curandState* rand_state, point3D* vertices, vector3D* normals, indx_struct* indices, int num_vertices, int num_normals, int num_shapes, int num_faces, int num_indices) {
	/**
	* Init scene with dragon model.
	*
	* @param[objects] pointer list that will contain the objects.
	* @param[world] pointer of the scene.
	* @param[cam] pointer of the cam.
	* @param[rand_state] CUDA rand state.
	* @param[vertices] mesh vertices.
	* @param[normals] mesh normals.
	* @param[indices] mesh indices.
	* @param[num_vertices] number of vertices.
	* @param[num_normals] number of normals.
	* @param[num_shapes] number of shapes.
	* @param[num_faces] number of faces.
	* @param[num_indices] number of indices.
	*/
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		material* red = new lambertian(new constant_texture(color(0.65f, 0.05f, 0.05f)));
		material* white = new lambertian(new constant_texture(color(0.73f, 0.73f, 0.73f)));
		material* green = new lambertian(new constant_texture(color(0.12f, 0.45f, 0.15f)));
		material* light = new diffuse_light(new constant_texture(color(2.0f, 2.0f, 2.0f)));

		// Light.
		object* ceiling_light = new rectangle(
			point3D(-10.0f, 10.0f, -10.0f),
			point3D(10.0f, 10.0f, -10.0f),
			point3D(-10.0f, 10.0f, 10.0f),
			point3D(10.0f, 10.0f, 10.0f)
		);
		instance* light_ptr = new instance(
			ceiling_light,
			light
		);
		*(objects) = light_ptr;

		object* sphere_model = new sphere();


		// Floor.
		instance* sphere_ptr = new instance(
			sphere_model,
			new lambertian(new checker_texture(new constant_texture(color(0.0f, 0.0f, 0.0f)), new constant_texture(color(1.0f, 1.0f, 1.0f))))
		);
		sphere_ptr->scale(100.0f, 100.0f, 100.0f);
		sphere_ptr->translate(0.0f, -100.5f, -1.0f);
		*(objects + 1) = sphere_ptr;

		// Mesh.
		object* mesh_model = new mesh(
			vertices,
			normals,
			indices,
			num_vertices,
			num_normals,
			num_shapes,
			num_faces,
			num_indices
		);

		instance* mesh_ptr = new instance(
			mesh_model,
			new dielectric(1.5f)
		);
		mesh_ptr->scale(0.04f, 0.04f, 0.04f);
		mesh_ptr->rotate_y(-90.0f);
		mesh_ptr->translate(0.0f, 1.0f, -1.0f);
		*(objects + 2) = mesh_ptr;

		if (USE_BVH == 1) {
			*world = new bvh(objects, 0, 3, rand_state);
		}
		else {
			*world = new object_list(objects, 3);
		}

		*cam = new camera(
			point3D(-10.0f, 10.0f, 8.0f),
			point3D(0.0f, 0.0f, -1.0f),
			vector3D(0.0f, 1.0f, 0.0f),
			30.0f,
			float(RES_X) / float(RES_Y)
		);
	}
}

__global__ void update_camera(camera** cam, float theta, float phi) {
	/**
	* Update camera look-from position based on mouse action.
	* 
	* @param[cam] pointer of the cam.
	* @param[theta] theta coordinate in spherical coordinates.
	* @param[phi] phi coordinate in spherical coordinates.
	*/
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		float radial_distance = magnitude(to_vector3D((*cam)->lookfrom - (*cam)->lookat));

		point3D lookfrom = point3D(
			radial_distance * sinf(theta) * sinf(phi),
			radial_distance * cosf(theta),
			radial_distance * sinf(theta) * cosf(phi)
		);
		lookfrom = lookfrom + (*cam)->lookat;

		(*cam)->update(lookfrom);
	}
}

__global__ void get_camera_info(camera** cam, point3D* init_look_from, float* init_radius) {
	/**
	* Update camera look-from position based on mouse action.
	*
	* @param[cam] pointer of the cam.
	* @param[init_look_from] initial camera look-from.
	* @param[init_radius] initial camera distance from look-from.
	*/
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*init_look_from = (*cam)->lookfrom;
		*init_radius = magnitude(to_vector3D((*cam)->lookfrom - (*cam)->lookat));
	}
}

__global__ void update_camera_for_animation(camera** cam, point3D look_from) {
	/**
	* Update camera look-from position based on mouse action.
	*
	* @param[cam] pointer of the cam.
	* @param[look_from] new camera look-from.
	*/
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		(*cam)->update(look_from);
	}
}

int main(int argc, char** argv) {
	if (init() == 1) {
		cout << "App Error! " << std::endl;
		return 1;
	}

	// Start time profiling.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float stop_watch = 0.0f;
	float total_time = 0.0f;

	int pixels = RES_X * RES_Y;

	// Allocate RES_X * RES_Y frame buffer.
	color* frame_buffer;
	HANDLE_CUDA_ERROR(cudaMallocManaged((void**)&frame_buffer, pixels * sizeof(color)));

	// Allocate random states.
	curandState* rand_state;
	curandState* rand_state_create_world;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&rand_state, pixels * sizeof(curandState)));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&rand_state_create_world, sizeof(curandState)));

	// Allocate create world random state.
	cudaEventRecord(start);
	rand_init<<<1, 1>>>(rand_state_create_world);
	cudaEventRecord(stop);
	HANDLE_CUDA_ERROR(cudaGetLastError());
	HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

	cudaEventElapsedTime(&stop_watch, start, stop);
	printf("Kernel function rand_init() took: %.5f seconds.\n", stop_watch * 0.001f);
	total_time += stop_watch * 0.001f;

	// Allocate objects.
	object** objects;
	if (CUDA_SCENE == 1) {
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&objects, 4 * sizeof(object*)));
	} else if (CUDA_SCENE == 2) {
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&objects, (22 * 22 + 1 + 3) * sizeof(object*)));
	}
	else if (CUDA_SCENE == 3) {
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&objects, 8 * sizeof(object*)));
	}
	else if (CUDA_SCENE == 4) {
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&objects, 3 * sizeof(object*)));
	}
	object** world;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&world, sizeof(object*)));
	camera** cam;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&cam, sizeof(camera*)));

	// Allocate mesh.
	int num_vertices;
	int num_normals;
	int num_shapes;
	int num_faces;
	int num_indices;

	point3D* vertices;
	vector3D* normals;
	indx_struct* indices;
	point3D* vertices_d;
	vector3D* normals_d;
	indx_struct* indices_d;
	if (CUDA_SCENE == 4) {
		load_mesh("../objs/dragon.obj", "../objs/", vertices, normals, indices, num_vertices, num_normals, num_shapes, num_faces, num_indices);

		printf("\nLoaded dragon.obj model \n");
		printf("Vertices: %d\n", num_vertices);
		printf("Normals: %d\n", num_normals);
		printf("Shapes: %d\n", num_shapes);
		printf("Faces: %d\n", num_faces);
		printf("Indices: %d\n\n", num_indices);

		cudaMallocManaged(&vertices_d, num_vertices * sizeof(point3D));
		cudaMemcpy(vertices_d, vertices, num_vertices * sizeof(point3D), cudaMemcpyHostToDevice);

		cudaMallocManaged(&normals_d, sizeof(vector3D));
		cudaMemcpy(normals_d, normals, sizeof(vector3D), cudaMemcpyHostToDevice);

		cudaMallocManaged(&indices_d, num_indices * sizeof(indx_struct));
		cudaMemcpy(indices_d, indices, num_indices * sizeof(indx_struct), cudaMemcpyHostToDevice);
	}

	cudaEventRecord(start);
	if (CUDA_SCENE == 1) {
		three_spheres<<<1, 1>>>(objects, world, cam, rand_state_create_world);
	}
	else if (CUDA_SCENE == 2) {
		random_spheres<<<1, 1>>>(objects, world, cam, rand_state_create_world);
	}
	else if (CUDA_SCENE == 3) {
		cornell_box<<<1, 1>>>(objects, world, cam, rand_state_create_world);
	}
	else if (CUDA_SCENE == 4) {
		dragon<<<1, 1>>>(objects, world, cam, rand_state_create_world, vertices_d, normals_d, indices_d, num_vertices, num_normals, num_shapes, num_faces, num_indices);
	}
	cudaEventRecord(stop);
	HANDLE_CUDA_ERROR(cudaGetLastError());
	HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

	cudaEventElapsedTime(&stop_watch, start, stop);
	printf("Kernel function for creation of scene %d took: %.5f seconds.\n", CUDA_SCENE, stop_watch * 0.001f);
	total_time += stop_watch * 0.001f;

	// Create blocks and threads.
	dim3 blocks(RES_X / THREAD_SIZE_X + 1, RES_Y / THREAD_SIZE_Y + 1);
	dim3 threads(THREAD_SIZE_X, THREAD_SIZE_Y);

	// Init render operation.
	cudaEventRecord(start);
	render_init<<<blocks, threads>>>(RES_X, RES_Y, rand_state);
	cudaEventRecord(stop);
	HANDLE_CUDA_ERROR(cudaGetLastError());
	HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

	cudaEventElapsedTime(&stop_watch, start, stop);
	printf("Kernel function render_init() took: %.5f seconds.\n", stop_watch * 0.001f);
	total_time += stop_watch * 0.001f;

	SDL_Event event;
	SDL_PollEvent(&event);
	if (ANIMATION == 1) {
		point3D init_look_from;
		point3D* init_look_from_dev;
		cudaMalloc(&init_look_from_dev, sizeof(point3D));

		float radius;
		float* radius_dev;
		cudaMalloc(&radius_dev, sizeof(float));
		
		get_camera_info<<<1, 1>>>(cam, init_look_from_dev, radius_dev);

		cudaMemcpy(&radius, radius_dev, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&init_look_from, init_look_from_dev, sizeof(point3D), cudaMemcpyDeviceToHost);
		cudaFree(init_look_from_dev);
		cudaFree(radius_dev);

		float theta_0 = atan2(init_look_from[1], init_look_from[0]) * RAD2DEG;

		int num_frame = 0;
		for (float theta = theta_0; theta < theta_0 + 361.0f; theta++) {
			float x = radius * cos(theta * DEG2RAD);
			float y = init_look_from[1];
			float z = radius * sin(theta * DEG2RAD);

			point3D look_from = point3D(x, y, z);

			update_camera_for_animation<<<1, 1>>>(cam, look_from);

			// Render allocated buffer.
			render<<<blocks, threads>>>(
				frame_buffer,
				RES_X,
				RES_Y,
				SPP,
				world,
				cam,
				rand_state
			);
			HANDLE_CUDA_ERROR(cudaGetLastError());
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			for (int j = ny - 1; j >= 0; j--) {
				for (int i = 0; i < nx; i++) {
					size_t pixel_index = j * nx + i;
					color c = frame_buffer[pixel_index];
					setPixel(i, j);
					setColor(c.r(), c.g(), c.b());
				}
			}

			SDL_RenderPresent(renderer);

			saveScreenshotBMP("../camera_animation/screenshot_" + to_string(num_frame) + ".bmp");
			cout << "Rendering of frame " + to_string(num_frame) + " of 360 completed." << endl;

			num_frame++;
		}
		cout << "Rendering completed." << endl;
	}
	else {
		// Render allocated buffer.
		cudaEventRecord(start);
		render<<<blocks, threads>>>(
			frame_buffer,
			RES_X,
			RES_Y,
			SPP,
			world,
			cam,
			rand_state
		);
		cudaEventRecord(stop);
		HANDLE_CUDA_ERROR(cudaGetLastError());
		HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&stop_watch, start, stop);
		printf("Kernel function render() took: %.5f seconds.\n", stop_watch * 0.001f);
		total_time += stop_watch * 0.001f;

		for (int j = ny - 1; j >= 0; j--) {
			for (int i = 0; i < nx; i++) {
				size_t pixel_index = j * nx + i;
				color c = frame_buffer[pixel_index];
				setPixel(i, j);
				setColor(c.r(), c.g(), c.b());
			}
		}

		SDL_RenderPresent(renderer);

		printf("\n*Total kernel execution time is %.5f seconds.\n", total_time);

		if (CUDA_SCENE != 4) {
			while (event.type != SDL_QUIT && event.key.keysym.sym != SDLK_q) {
				poll_event(event);

				if (mouse_drag && MOUSE_DRAG) {
					update_camera << <1, 1 >> > (cam, theta, phi);
					HANDLE_CUDA_ERROR(cudaGetLastError());
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

					// Render allocated buffer.
					render << <blocks, threads >> > (
						frame_buffer,
						RES_X,
						RES_Y,
						SPP,
						world,
						cam,
						rand_state
						);
					HANDLE_CUDA_ERROR(cudaGetLastError());
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

					for (int j = ny - 1; j >= 0; j--) {
						for (int i = 0; i < nx; i++) {
							size_t pixel_index = j * nx + i;
							color c = frame_buffer[pixel_index];
							setPixel(i, j);
							setColor(c.r(), c.g(), c.b());
						}
					}

					SDL_RenderPresent(renderer);
				}
			}
		}
		else {
			saveScreenshotBMP("SCREEN.bmp");
		}
	}

	// Destroy created event.
	HANDLE_CUDA_ERROR(cudaEventDestroy(start));
	HANDLE_CUDA_ERROR(cudaEventDestroy(stop));
	// Sync and deallocate pointers.
	HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
	HANDLE_CUDA_ERROR(cudaGetLastError());

	HANDLE_CUDA_ERROR(cudaFree(rand_state_create_world));
	HANDLE_CUDA_ERROR(cudaFree(rand_state));

	HANDLE_CUDA_ERROR(cudaFree(frame_buffer));

	HANDLE_CUDA_ERROR(cudaFree(objects));
	HANDLE_CUDA_ERROR(cudaFree(world));

	if (CUDA_SCENE == 4) {
		HANDLE_CUDA_ERROR(cudaFree(vertices_d));
		HANDLE_CUDA_ERROR(cudaFree(normals_d));
		HANDLE_CUDA_ERROR(cudaFree(indices_d));
	}

	close();
	return 0;
}