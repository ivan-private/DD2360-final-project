#pragma once

#include "vec3.h"
#include "material.h"


struct HitRecord
{
    float t;
    vec3 p;
    vec3 normal;
    Material mat;
};
