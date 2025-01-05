#pragma once

#include "hitable.h"



class BVH_Node 
{
public:
    __device__ 
    BVH_Node(Hitable* l, int n, int depth=0)
    {
        if (n == 1) 
        {
            left = right = nullptr;
            hitable = l[0];
        } 
        // else if (n == 2) 
        // {
        //     left = &l[0];
        //     right = &l[1];
        //     bb = BoundingBox::surrounding_box(left->bounding_box(), right->bounding_box());
        // } 
        else 
        {
            int axis = depth % 3; // Cycle through x, y, z axes
            int mid = n / 2;
            
            // Partition list based on axis
            sort(l, n, axis);
            

            left = new BVH_Node(l, mid, depth + 1);
            right = new BVH_Node(&l[mid], n - mid, depth + 1);
            hitable = BoundingBox::surrounding_box(left->bounding_box(), right->bounding_box());
        }
    }

    __device__ 
    ~BVH_Node() 
    {
        if (left) delete left;
        if (right) delete right;
    }



    __device__ 
    bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const  
    {
        if (!hitable.hit(r, t_min, t_max, rec)) 
        {
            return false;
        }

        // maybe left and right are null?? 
        bool hit_left = left->hit(r, t_min, t_max, rec);
        bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

        return hit_left || hit_right;
    }


    __device__ 
    BoundingBox bounding_box() const  
    {
        return hitable.bounding_box();
    }


private:
    __device__ 
    static void sort(Hitable* l, int n, int axis) 
    {
        for (int i = 0; i < n - 1; ++i) 
        {
            for (int j = i + 1; j < n; ++j) 
            {
                if (l[i].bounding_box().lower[axis] > l[j].bounding_box().lower[axis]) 
                {
                    Hitable temp = l[i];
                    l[i] = l[j];
                    l[j] = temp;
                }
            }
        }

    }


private:
    BVH_Node* left;
    BVH_Node* right;
    Hitable hitable = BoundingBox();
};


class BVH
{
public:
    
    __device__ 
    BVH(Hitable* list, int n) : root(list, n) {}


    __device__ 
    bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const  
    {
        return root.hit(r, t_min, t_max, rec);
    }

private: 
    BVH_Node root;

};






