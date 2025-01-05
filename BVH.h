#ifndef BVH_H
#define BVH_H

#include "hitable.h"
#include "bounding_box.h" 
#include <algorithm>

struct bvh_node : hitable
{
    hitable* left;
    hitable* right;
    BoundingBox bb;

    __device__ bvh_node(hitable** l, int n, int depth=0)
    {
        if (n == 1) {
            //printf("N == 1\n");
            left = right = l[0];
            bb = l[0]->bounding_box();
        } else if (n == 2) {
            //printf("N == 2\n");
            left = l[0];
            right = l[1];
            bb = BoundingBox::surrounding_box(l[0]->bounding_box(), l[1]->bounding_box());
        } else {
            int axis = depth % 3; // Cycle through x, y, z axes
            int mid = n / 2;
            
            // Partition list based on axis
            sort(l, n, axis);
            
            // Split into left and right subtrees
            //printf("1\n");
            left = new bvh_node(l, mid, depth + 1);
            //printf("2\n");
            right = new bvh_node(&l[mid], n - mid, depth + 1);
            //printf("3\n");
            bb = BoundingBox::surrounding_box(left->bounding_box(), right->bounding_box());
            //printf("4\n");
        }
    }


    __device__ static void sort(hitable** l, int n, int axis) 
    {
        //printf("BEFORE SORT\n");
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                //printf("IN LOOP i = %d, j = %d\n", i, j);
                //printf("l[i] memory = %p\n", &l[i]);
                //printf("l[j] memory = %p\n", &l[j]);

                if (l[i] == nullptr) {
                    //printf("l[i] is null for i == %d\n", i);
                }
                if (l[j] == nullptr) {
                    //printf("l[j] is null for j == %d\n", j);
                }
                if (axis > 2) {
                    //printf("AXIS TO BIG = %d\n", axis);
                }
                if (l[i]->bounding_box().lower[axis] > l[j]->bounding_box().lower[axis]) {
                    //printf("IF WAS TRUE\n");
                    hitable* temp = l[i];
                    l[i] = l[j];
                    l[j] = temp;
                }
                else
                {
                    //printf("IF WAS FALSE\n");
                }
            }
        }

    }


    __device__ ~bvh_node()
    {
        // might delete a leaf object, which is a sphere
        // this is not allowed
        if (left) delete left;
        if (right) delete right;
    }


    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override 
    {
        if (!bb.hit(r, t_min, t_max, rec)) 
        {
            return false;
        }

        // maybe left and right are null?? 
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
    __device__ BVH(hitable** list, int n) : root(list, n)
    {
    }


    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override 
    {
        return root.hit(r, t_min, t_max, rec);
    }

    __device__ BoundingBox bounding_box() const override 
    {
        return root.bb;
    }

public: // TODO: should be private, public now for debugging
    bvh_node root;

};






#endif
