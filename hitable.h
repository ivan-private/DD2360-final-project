#pragma once

#include <cassert>

#include "ray.h"
#include "sphere.h"
#include "bounding_box.h"
#include "hit_record.h"


class Hitable
{
public:
    
    enum class HitableType { none, sphere, bounding_box };

    // __device__ 
    // Hitable() : type(HitableType::none), bb(BoundingBox()) {}

    // __device__
    // Hitable(const Hitable& other) = default;


    // __device__
    // Hitable(Hitable&& other) = default;


    __device__
    Hitable& operator = (const Hitable& other) 
    {
        type = other.type;
        switch (type)
        {
            case HitableType::sphere:
                s = other.s;
                break;

            case HitableType::bounding_box: 
                bb = other.bb;
                break;
            
            default:
                assert(false);
        }

        return *this;
    }

    
    template<typename T>
    __device__
    Hitable(T hitable_object) 
    {
        if constexpr (std::is_same_v<T, Sphere>)
        {
            type = HitableType::sphere;
            s = hitable_object;
        }
        else if constexpr (std::is_same_v<T, BoundingBox>)
        {
            type = HitableType::bounding_box;
            bb = hitable_object;
        }
        else 
        {
            assert(false);
        }
    }

    template<typename T>
    __device__
    Hitable& operator = (T hitable_object) 
    {
        if constexpr (std::is_same_v<T, Sphere>)
        {
            type = HitableType::sphere;
            s = hitable_object;
        }
        else if constexpr (std::is_same_v<T, BoundingBox>)
        {
            type = HitableType::bounding_box;
            bb = hitable_object;
        }
        else 
        {
            assert(false);
        }

        return *this;
    }


    __device__ 
    bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        switch (type)
        {
            case HitableType::sphere: 
                return s.hit(r, t_min, t_max, rec);

            case HitableType::bounding_box: 
                return bb.hit(r, t_min, t_max, rec);
        }

        // assert(false);
        return false;
    }

    __device__
    BoundingBox bounding_box() const
    {
        switch (type)
        {
            case HitableType::sphere: 
                return s.bounding_box();

            case HitableType::bounding_box: 
                return bb.bounding_box();
        }

        assert(false);
        return BoundingBox();
    }


private:
    Hitable::HitableType type = Hitable::HitableType::none;
    union 
    {
        Sphere s;
        BoundingBox bb;
    };

};
