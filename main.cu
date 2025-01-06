#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#define thread_num_x 4
#define thread_num_y 4

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) { // bottleneck of this function
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, int ns, curandState *rand_state) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = tx + blockIdx.x * blockDim.x;
    int j = ty + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curandState local_rand_state;
    curand_init(1984+pixel_index, 0, 0, &local_rand_state);

    // // To make sure &local_rand_state is in correct value in each sample
    // rand_state[pixel_index * ns] = local_rand_state;
    // for (int k = 1; k < ns; k++){
    //     curand_uniform(&local_rand_state);
    //     curand_uniform(&local_rand_state);
    //     random_in_unit_disk(&local_rand_state);
    //     random_in_unit_sphere(&local_rand_state);
    //     rand_state[pixel_index*ns + k] = local_rand_state;
    // }

    // Temporary buffer to store random states
    __shared__ curandState rand_buffer[8][8][10];

    // Store the first random state directly
    rand_buffer[tx][ty][0] = local_rand_state;
    
    // Process all subsequent random states in a temporary buffer
    for (int k = 1; k < ns; k++) {
        // Update the random state locally
        curand_uniform(&local_rand_state);
        curand_uniform(&local_rand_state);
        random_in_unit_disk(&local_rand_state);
        random_in_unit_sphere(&local_rand_state); // This one is problematic. Not all the sample will do this actually
        rand_buffer[tx][ty][k]  = local_rand_state;
    }

    // Write all states back to global memory
    for (int k = 0; k < ns; k++) {
        rand_state[pixel_index*ns + k] = rand_buffer[tx][ty][k];
    }
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int s = threadIdx.z;

    if((i >= max_x) || (j >= max_y) || (s >= ns)) return;

    __shared__ vec3 shared_color[thread_num_x][thread_num_y]; // Have to modify this parameter
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (s == 0) {
        shared_color[ty][tx] = vec3(0.0, 0.0, 0.0);
    }
    __syncthreads();

    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index*ns + s];

 
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    vec3 sample_color = color(r, world, &local_rand_state);

    // atomicAdd(&fb[pixel_index].e[0], sample_color.e[0]);
    // atomicAdd(&fb[pixel_index].e[1], sample_color.e[1]);
    // atomicAdd(&fb[pixel_index].e[2], sample_color.e[2]);

    atomicAdd(&shared_color[ty][tx].e[0], sample_color.e[0]);
    atomicAdd(&shared_color[ty][tx].e[1], sample_color.e[1]);
    atomicAdd(&shared_color[ty][tx].e[2], sample_color.e[2]);
    __syncthreads();

    if (s == ns - 1) {
        atomicAdd(&fb[pixel_index].e[0], shared_color[ty][tx].e[0]);
        atomicAdd(&fb[pixel_index].e[1], shared_color[ty][tx].e[1]);
        atomicAdd(&fb[pixel_index].e[2], shared_color[ty][tx].e[2]);
    }
}

__global__ void normalize(vec3 *fb, int max_x, int max_y, int ns) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    fb[pixel_index] /= float(ns);
    fb[pixel_index].e[0] = sqrt(fb[pixel_index].e[0]);
    fb[pixel_index].e[1] = sqrt(fb[pixel_index].e[1]);
    fb[pixel_index].e[2] = sqrt(fb[pixel_index].e[2]);
}


#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 16;
    int ty = 16;

    // std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    // std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    int thread_x_max = 10;
    int thread_y_max = 10;
    int best_thread_x_config, best_thread_y_config;
    double best_spent_time = 100;

    for(int thread_x = thread_num_x; thread_x < thread_num_x+1; thread_x++){
        for(int thread_y = thread_num_y; thread_y < thread_num_y+1; thread_y++){
            // allocate FB
            vec3 *fb;
            checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

            // allocate random state
            curandState *d_rand_state;
            checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*ns*sizeof(curandState)));
            curandState *d_rand_state2;
            checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

            // we need that 2nd random state to be initialized for the world creation
            rand_init<<<1,1>>>(d_rand_state2);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // make our world of hitables & the camera
            hitable **d_list;
            int num_hitables = 22*22+1+3;
            checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
            hitable **d_world;
            checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
            camera **d_camera;
            checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
            create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            clock_t start, stop;
            start = clock();
            // Render our buffer
            render_init<<<dim3((nx+7)/8, (ny+7)/8), dim3(8, 8)>>>(nx, ny, ns, d_rand_state);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            render<<<dim3((nx+thread_x-1)/thread_x, (ny+thread_y-1)/thread_y), dim3(thread_x, thread_y, 10)>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            normalize<<<dim3((nx+15)/16, (ny+15)/16), dim3(16, 16)>>>(fb, nx, ny, ns);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            stop = clock();
            double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

            if (timer_seconds < best_spent_time){
                best_thread_x_config = thread_x;
                best_thread_y_config = thread_y;
                best_spent_time = timer_seconds;
            }
            // Output FB as Image
            // std::cout << "P3\n" << nx << " " << ny << "\n255\n";
            // for (int j = ny-1; j >= 0; j--) {
            //     for (int i = 0; i < nx; i++) {
            //         size_t pixel_index = j*nx + i;
            //         int ir = int(255.99*fb[pixel_index].r());
            //         int ig = int(255.99*fb[pixel_index].g());
            //         int ib = int(255.99*fb[pixel_index].b());
            //         std::cout << ir << " " << ig << " " << ib << "\n";
            //     }
            // }


            // clean up
            checkCudaErrors(cudaDeviceSynchronize());
            free_world<<<1,1>>>(d_list,d_world,d_camera);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaFree(d_camera));
            checkCudaErrors(cudaFree(d_world));
            checkCudaErrors(cudaFree(d_list));
            checkCudaErrors(cudaFree(d_rand_state));
            checkCudaErrors(cudaFree(d_rand_state2));
            checkCudaErrors(cudaFree(fb));

            cudaDeviceReset();
        }
    }
    std::cerr << "thread_x: " << best_thread_x_config << "\n" << "thread_y: " << best_thread_y_config << "\n";
    std::cerr << "Render took " << best_spent_time << " seconds.\n\n";
}
