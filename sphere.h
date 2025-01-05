#pragma once

#include <cassert>

#include "bounding_box.h"
#include "vec3.h"
#include "ray.h"
#include "material.h"
#include "hit_record.h"

struct Sphere  
{
    __host__ __device__ Sphere() {}

    __host__ __device__ 
    Sphere(vec3 cen, float r, Material m) : center(cen), radius(r), mat(m)  {};

    __device__ 
    bool hit(const ray& r, float tmin, float tmax, HitRecord& rec) const;

    __host__ __device__ 
    BoundingBox bounding_box() const;

    vec3 center;
    float radius;
    Material mat;
};

// try to optimize this
__device__ bool Sphere::hit(const ray& r, float t_min, float t_max, HitRecord& rec) const 
{
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) 
    {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat = mat;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat = mat;
            return true;
        }
    }
    return false;
}


__device__ BoundingBox Sphere::bounding_box() const
{
    const vec3 r(radius, radius, radius);
    return BoundingBox(center - r, center + r);
}

