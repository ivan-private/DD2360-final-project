#ifndef BVH_H
#define BVH_H

#include "hitable.h"
#include "bounding_box.h" 
#include <algorithm>

struct bvh_node : public hitable 
{
    hitable* left;
    hitable* right;
    BoundingBox bb;


    __device__ bvh_node(hitable** l, int n, int depth = 0) 
    {
        if (n == 1) {
            left = right = l[0];
            bb = l[0]->bounding_box();
        } else if (n == 2) {
            left = l[0];
            right = l[1];
            bb = BoundingBox::surrounding_box(l[0]->bounding_box(), l[1]->bounding_box());
        } else {
            int axis = depth % 3; // Cycle through x, y, z axes
            int mid = n / 2;

            // Partition list based on axis
            for (int i = 0; i < n - 1; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (l[i]->bounding_box().lower[axis] > l[j]->bounding_box().lower[axis]) {
                        hitable* temp = l[i];
                        l[i] = l[j];
                        l[j] = temp;
                    }
                }
            }

            // Split into left and right subtrees
            left = new bvh_node(l, mid, depth + 1);
            right = new bvh_node(l + mid, n - mid, depth + 1);
            bb = BoundingBox::surrounding_box(left->bounding_box(), right->bounding_box());
        }
    }

    __device__ ~bvh_node()
    {
        if (left) delete left;
        if (right) delete right;
    }


    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override 
    {
        if (!bb.hit(r, t_min, t_max, rec)) 
        {
            return false;
        }

        bool hit_left = left->hit(r, t_min, t_max, rec);
        bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

        return hit_left || hit_right;
    }


    __device__ BoundingBox bounding_box() const override 
    {
        return bb;
    }
};



class BVH : public hitable {
public:
    __device__ BVH(hitable** list, int n) : root(list, n) {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override 
    {
        return root.hit(r, t_min, t_max, rec);
    }

    __device__ BoundingBox bounding_box() const override 
    {
        return root.bb;
    }

public:
    bvh_node root;
};






#endif
