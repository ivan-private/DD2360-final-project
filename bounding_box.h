#pragma once

#include "hitable.h"
#include "vec3.h"


struct BoundingBox : public hitable
{
    vec3 lower;
    vec3 upper;


    __device__ BoundingBox() {}
    __device__ BoundingBox(vec3 lower, vec3 upper)
        : lower(lower), upper(upper) {}

    __device__ BoundingBox bounding_box() const override { return *this; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const float invD = 1.0f / r.direction()[axis];
            float t0 = (lower[axis] - r.origin()[axis]) * invD;
            float t1 = (upper[axis] - r.origin()[axis]) * invD;

            if (invD < 0.0f)  // Swap t0 and t1 if the ray is going in the negative direction
            {
                const float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            t_min = max(t0, t_min);  // Update t_min to the furthest minimum
            t_max = min(t1, t_max);  // Update t_max to the closest maximum

            // if (t_max <= t_min)  // No intersection if the intervals do not overlap
            //     return false;
        }
        
        return t_min < t_max;
    }


    // Helper function to compute surrounding bounding box
    __device__ static BoundingBox surrounding_box(const BoundingBox& box0, const BoundingBox& box1) 
    {
        const vec3 small(min(box0.lower.x(), box1.lower.x()),
                min(box0.lower.y(), box1.lower.y()),
                min(box0.lower.z(), box1.lower.z()));

        const vec3 big(max(box0.upper.x(), box1.upper.x()),
                max(box0.upper.y(), box1.upper.y()),
                max(box0.upper.z(), box1.upper.z()));

        return BoundingBox(small, big);
}
};


