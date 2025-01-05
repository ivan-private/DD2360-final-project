#pragma once

#include <concepts>

#include "ray.h"
#include "hit_record.h"
#include "bounding_box.h"

// namespace traits
// {

//     template <typename T>
//     concept Hitable = requires(T x, const ray& r, float t_min, float t_max, hit_record& rec)
//     {
//         { x.hit(r, t_min, t_max, rec) } -> std::same_as<bool>;

//         { x.bounding_box() } -> std::same_as<BoundingBox>;
//     };


// } // namespace traits