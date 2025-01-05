#pragma once

#include "hitable.h"

class HitableList
{
public:
    __device__ 
    HitableList(Sphere* list, int n) : list(list), list_size(n) {}

    __device__ 
    bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) 
        {
            if (list[i].hit(r, t_min, closest_so_far, temp_rec)) 
            {
                hit_anything = true; 
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }


private:
    Sphere* list;
    int list_size;
};
