#pragma once
#include <curand_kernel.h>
#include <cassert>

#include "ray.h"


__device__ 
float schlick(float cosine, float ref_idx) 
{
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}


__device__ 
bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) 
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) 
    {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    
    return false;
}


__device__ 
vec3 random_in_unit_sphere(curandState *local_rand_state) 
{
    vec3 p;
    do {
        const vec3 random_vector = vec3(curand_uniform(local_rand_state),
                                        curand_uniform(local_rand_state),
                                        curand_uniform(local_rand_state));

        p = 2.0f*random_vector - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ 
vec3 reflect(const vec3& v, const vec3& n) 
{
     return v - 2.0f*dot(v,n)*n;
}



struct Lambertian
{
    __host__ __device__ 
    Lambertian(const vec3& a) : albedo(a) {}


    __device__  
    bool scatter(const ray& r_in, const vec3& p, const vec3& normal, vec3& attenuation, ray& scattered, 
        curandState *local_rand_state) const  
    {
            vec3 target = p + normal + random_in_unit_sphere(local_rand_state);
            scattered = ray(p, target-p);
            attenuation = albedo;
            return true;
    }


    vec3 albedo;
};



struct Metal
{
        __host__ __device__ 
        Metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }


        __device__
        bool scatter(const ray& r_in, const vec3& p, const vec3& normal, vec3& attenuation, ray& scattered, 
            curandState *local_rand_state) const  
        {
            vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
            scattered = ray(p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), normal) > 0.0f);
        }



        vec3 albedo;
        float fuzz;
};



struct Dielectric 
{

    __host__ __device__ 
    Dielectric(float ri) : ref_idx(ri) {}


    __device__ 
    bool scatter(const ray& r_in, const vec3& p, const vec3& normal, vec3& attenuation, ray& scattered,
        curandState *local_rand_state) const  
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), normal) > 0.0f) {
            outward_normal = -normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(p, reflected);
        else
            scattered = ray(p, refracted);
        return true;
    }


    float ref_idx;
};




class Material
{
public:
    enum class MaterialType {none, lambertian, metal, dielectric};

    __host__ __device__
    Material() : type(MaterialType::none) {}

    template<typename T>
    __host__ __device__
    Material(T material)
    {
        if constexpr (std::is_same_v<T, Lambertian>)
        {
            type = MaterialType::lambertian;
            lambertian = material;
        }
        else if constexpr (std::is_same_v<T, Metal>)
        {
            type = MaterialType::metal;
            metal = material;
        }
        else if constexpr (std::is_same_v<T, Dielectric>)
        {
            type = MaterialType::dielectric;
            dielectric = material;
        }
        else 
        {
            assert(false);
        }
    }


    template<typename T>
    __host__ __device__
    T operator = (T material)
    {
        if constexpr (std::is_same_v<T, Lambertian>)
        {
            type = MaterialType::lambertian;
            lambertian = material;
        }
        else if constexpr (std::is_same_v<T, Metal>)
        {
            type = MaterialType::metal;
            metal = material;
        }
        else if constexpr (std::is_same_v<T, Dielectric>)
        {
            type = MaterialType::dielectric;
            dielectric = material;
        }
        else 
        {
            assert(false);
        }

        return *this;
    }

    __device__ 
    bool scatter(const ray& r_in, const vec3& p, const vec3& normal, vec3& attenuation, ray& scattered, 
        curandState *local_rand_state) const 
    {
        switch (type)
        {
            case Material::MaterialType::lambertian:
                return lambertian.scatter(r_in, p, normal, attenuation, scattered, local_rand_state);
            
            case Material::MaterialType::metal:
                return metal.scatter(r_in, p, normal, attenuation, scattered, local_rand_state);

            case Material::MaterialType::dielectric:
                return dielectric.scatter(r_in, p, normal, attenuation, scattered, local_rand_state);

            // default:
            //     printf("\nASSERT FROM Material::scatter. Type == %d", type);
            //     assert(false);
        }

        return false;
    }


    __device__ 
    operator bool()
    {
        return type != Material::MaterialType::none;
    }


public:
    Material::MaterialType type;
    union 
    {
        Lambertian lambertian;
        Metal metal;
        Dielectric dielectric;
    };
};