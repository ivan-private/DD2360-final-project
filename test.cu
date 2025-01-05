#include <iostream>

#include <concepts>

#include <variant>
#include <vector>

#include "sphere.h"
#include "hitable.h"

// using DataType = std::variant<IntWrapper, DoubleWrapper>;

// struct IntWrapper
// {
//     int data;

//     __host__ __device__ IntWrapper(int data) : data(data) {}

//     __host__ __device__  bool is_zero() { return data == 0; }
// };

// struct DoubleWrapper
// {
//     double data;

//     __host__ __device__ DoubleWrapper(double data) : data(data) {}

//     __host__ __device__  bool is_zero() { return data < 1e-9; }
// };


// template<typename T> 
// concept NumberWrapper = requires (T x)
// {
//     requires std::is_arithmetic_v<decltype(x.data)>;
//     { x.is_zero() } -> std::same_as<bool>;
// };


// struct IsZeroVisitor
// {
//     __host__ __device__ bool operator()(IntWrapper wrapper) { return wrapper.is_zero(); }
//     __host__ __device__ bool operator()(DoubleWrapper wrapper) { return wrapper.is_zero(); }
// };


// struct GetDataVisitor
// {
//     __host__ __device__ int operator()(const IntWrapper& wrapper) { return wrapper.data; }
//     __host__ __device__ double operator()(const DoubleWrapper& wrapper) { return wrapper.data; }
// };


// template <NumberWrapper T>
// __device__ bool is_zero(const T& x)
// {
//     return x.is_zero();
// }

// __global__ void kernel(DataType* x, int n)
// {
//     if (threadIdx.x == 0 && blockIdx.x == 0)
//     {
//         for (int i = 0; i < n; i++)
//         {
            
//             printf("Kernel call worked!\n");
//             printf("DataType = %s\n", std::holds_alternative<IntWrapper>(x[i]) ? "IntWrapper" : "DoubleWrapper");

//             bool ans = is_zero(x[i]);
//             printf("Is zero = %s\n", ans? "true" : "false");
//             //printf("data = %.2f\n", std::visit(IsZeroVisitor{}, x[i]));
//         }
        
//     }
// }




// struct BoundingBox : public hitable<BoundingBox>
// {
//     vec3 lower;
//     vec3 upper;


//     __host__ __device__ 
//     BoundingBox() {}


//     __host__ __device__ 
//     BoundingBox(vec3 lower, vec3 upper)
//         : lower(lower), upper(upper) {}


//     __host__ __device__ 
//     const BoundingBox& bounding_box() const { return *this; }


//     __device__
//     bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const 
//     {
//         for (int axis = 0; axis < 3; ++axis)
//         {
//             const float invD = 1.0f / r.direction()[axis];
//             float t0 = (lower[axis] - r.origin()[axis]) * invD;
//             float t1 = (upper[axis] - r.origin()[axis]) * invD;

//             if (invD < 0.0f)  // Swap t0 and t1 if the ray is going in the negative direction
//             {
//                 const float temp = t0;
//                 t0 = t1;
//                 t1 = temp;
//             }

//             t_min = max(t0, t_min);  // Update t_min to the furthest minimum
//             t_max = min(t1, t_max);  // Update t_max to the closest maximum

//             // if (t_max <= t_min)  // No intersection if the intervals do not overlap
//             //     return false;
//         }
        
//         return t_min < t_max;
//     }


//     // Helper function to compute surrounding bounding box
//     __host__ __device__
//     static BoundingBox surrounding_box(const BoundingBox& box0, const BoundingBox& box1) 
//     {
//         const vec3 small(min(box0.lower.x(), box1.lower.x()),
//                 min(box0.lower.y(), box1.lower.y()),
//                 min(box0.lower.z(), box1.lower.z()));

//         const vec3 big(max(box0.upper.x(), box1.upper.x()),
//                 max(box0.upper.y(), box1.upper.y()),
//                 max(box0.upper.z(), box1.upper.z()));

//         return BoundingBox(small, big);
//     }
// };


// using HitableType = std::variant<BoundingBox>;

// struct PrintType
// {
//     __host__ __device__
//     void operator()(const BoundingBox& bb) { printf("BoundingBox\n"); }
// };

// __global__
// void kernel(HitableType* list, int n)
// {
//     if ( !(threadIdx.x == 0 && blockIdx.x == 0) ) return;

//     for (int i = 0; i < n; i++) 
//     {
//         printf("Type of list[%d] = ", i);
//         std::visit(PrintType{}, list[i]);

//         // const BoundingBox& bb = list[i].bounding_box();
//         // printf("%d. Bounding box = {%.1f, %.1f, %.1f} - {%.1f, %.1f, %.1f}\n", 
//         // i, bb.lower.x(), bb.lower.y(), bb.lower.z(), bb.upper.x(), bb.upper.y(), bb.upper.z());

//     }
// }

__global__
void create_world_gpu(Sphere* list)
{
    if ( !(threadIdx.x == 0 && blockIdx.x == 0) ) return;

    list[0] = Sphere(vec3(0, 1,0),  1.0,  Dielectric(1.5));
    list[1] = Sphere(vec3(-4, 1, 0), 1.0, Lambertian(vec3(0.4, 0.2, 0.1)));
    list[2] = Sphere(vec3(4, 1, 0),  1.0, Metal(vec3(0.7, 0.6, 0.5), 0.0));
}


__global__
void create_world_gpu(Hitable* list)
{
    if ( !(threadIdx.x == 0 && blockIdx.x == 0) ) return;

    list[0] = Sphere(vec3(0, 1,0),  1.0,  Dielectric(1.5));
    list[1] = Sphere(vec3(-4, 1, 0), 1.0, Lambertian(vec3(0.4, 0.2, 0.1)));
    list[2] = Sphere(vec3(4, 1, 0),  1.0, Metal(vec3(0.7, 0.6, 0.5), 0.0));
}


void create_world_cpu(Sphere* list)
{
    list[0] = Sphere(vec3(0, 1,0),  1.0,  Dielectric(1.5));
    list[1] = Sphere(vec3(-4, 1, 0), 1.0, Lambertian(vec3(0.4, 0.2, 0.1)));
    list[2] = Sphere(vec3(4, 1, 0),  1.0, Metal(vec3(0.7, 0.6, 0.5), 0.0));
}

void create_world_cpu(Hitable* list)
{
    list[0] = Sphere(vec3(0, 1,0),  1.0,  Dielectric(1.5));
    list[1] = Sphere(vec3(-4, 1, 0), 1.0, Lambertian(vec3(0.4, 0.2, 0.1)));
    list[2] = Sphere(vec3(4, 1, 0),  1.0, Metal(vec3(0.7, 0.6, 0.5), 0.0));
}


void print_spheres(Sphere* list, int n)
{
    for (int i = 0; i < n; i++)
    {
        const Sphere& s = list[i];
        std::cout << i+1 << ". {" << s.center << "}, r = " << s.radius << "\n";
    }
}

void print_world(Hitable* list, int n)
{
    for (int i = 0; i < n; i++)
    {
        const BoundingBox& bb = list[i].bounding_box();
        std::cout << i+1 << ". {" << bb.lower << "}, {" << bb.upper << "}" << "\n";
    }
}


int main()
{

    std::cout << "sizeof(Sphere) = " << sizeof(Sphere) << '\n';
    std::cout << "sizeof(Hitable) = " << sizeof(Hitable) << '\n';

    const int n = 3;

    Sphere* spheres; cudaMallocManaged(&spheres, n * sizeof(Sphere));

    //create_world_gpu<<<1,1>>>(list);
    create_world_cpu(spheres);
    print_spheres(spheres, n);


    Hitable* list; cudaMallocManaged(&list, n * sizeof(Hitable));
    create_world_gpu<<<1,1>>>(list);
    cudaDeviceSynchronize();
    print_world(list, n);

    cudaDeviceSynchronize();
}