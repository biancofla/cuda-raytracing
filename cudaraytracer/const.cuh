#define RES_X 1366	
#define RES_Y 768

// Samples per pixel.
#define SPP 100

#define RAY_BOUNCES 12

#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8	

// Math constants.
#define M_PI 3.14159265358979323846f
#define M_PI_2 1.57079632679489661923f
#define DEG2RAD M_PI / 180.0f
#define RAD2DEG 180.0f / M_PI
#define EPSILON 0.0000001f

#define RND_VEC vector3D(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))
#define RND (curand_uniform(&local_rand_state))

/**
* Scenes:
* 1. Three spheres;
* 2. Multiple random spheres with random materials;
* 3. Cornell box.
* 4. Dragon.
*/
#define CUDA_SCENE 2

// Choose 1 if you want to enable BVH.
#define USE_BVH 0

// Enable mouse movements.
#define MOUSE_DRAG ((CUDA_SCENE == 1 && SPP < 101) || (CUDA_SCENE == 2 && SPP == 1) || (CUDA_SCENE == 3 && SPP < 6))

// Enable animation.
#define ANIMATION 0