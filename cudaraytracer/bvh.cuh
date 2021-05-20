#pragma once

#include <iostream>

#include "object_list.cuh"
#include "object.cuh"

#include "thrust/sort.h"

class bvh : public object {
public:
	bvh() = default;
	__device__ bvh::bvh(object** objects, int start, int end, curandState* rand_state);

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

	__device__ virtual bool bounding_box(aabb& output_box) const;

	object* left;
	object* right;
	aabb box;
};

struct bvh_node {
    size_t s, e;
    bvh* bvh;
};

__device__ bool bvh::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!box.hit(r, t_min, t_max)) {
        return false;
    }
    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);
    return hit_left || hit_right;
}

__device__ bool bvh::bounding_box(aabb& output_box) const {
    output_box = box;
    return true;
}

__device__ bvh::bvh(object** objects, int start, int end, curandState* rand_state) {
    int axis = 0;

    // Define a comparator function.
    auto comparator = [&axis] __device__(object * a, object * b) {
        aabb box_left, box_right;

        if (!a->bounding_box(box_left) || !b->bounding_box(box_right)) {
            printf("%s\n", "no bounding box in bvh_node constructor");
        }

        return box_left.pmin[axis] < box_right.pmin[axis];
    };

    // Post-order traversal stack-pushing strategy.
    auto next = [this, objects, rand_state, comparator, &axis] __device__(bvh_node n, bvh_node* stack, int& i) {
        while (true) {
            // Remaining objects to evaluate.
            int objects_remained = n.e - n.s;
            if (objects_remained > 2) {
                // Choose randomly an axis.
                axis = ceilf(curand_uniform(rand_state) * 3);
                // Sort objects with a comparator (defined as a lambda ex-
                // pression in the previous rows of code).
                thrust::sort(objects + n.s, objects + n.e, comparator);
                // Define a aplit strategy. We split up the nodes by two.
                int mid_element = n.s + objects_remained / 2;
                // Init right and left nodes of n.
                n.bvh->right = new bvh(); n.bvh->left = new bvh();
                // Push to the stack the placeholder node that will contain
                // the objects in range (mid_element; n.e].
                stack[++i] = { mid_element, n.e, ((bvh*)n.bvh->right) };
                // Add to the stack the parent node n.
                stack[++i] = n;
                // Set bvh_node n as the placeholder node containing the
                // objects in range [n.s; mid_element).
                n = { n.s, mid_element, ((bvh*)n.bvh->left) };
            } else {
                // Push to the stack the last remaining element.
                stack[++i] = n;
                break;
            }
        }
    };

    bvh_node* stack = new bvh_node[end * 2];
    int i = -1;
    // After the execution of the lambda expression, the variable 
    // i will assume the value of the existing nodes in the stack.
    next({ start, end, this }, stack, i);

    while (i >= 0) {
        // We pop up the nodes one by one.
        auto n = stack[i--];
        // If the popped item has a right child and the this right
        // child is at top of the stack, then we remove the right 
        // child from the stack, push the root back and set root 
        // as root's right child.
        if (i >= 0 && stack[i].bvh == n.bvh->right) {
            auto right_node = stack[i];
            stack[i] = n;
            next(right_node, stack, i);
            n = stack[i--];
        }
        // Remaining objects to evaluate.
        int objects_remained = n.e - n.s;
        switch (objects_remained) {
        case 1: {
            n.bvh->left = n.bvh->right = objects[n.s];
            break;
        }
        case 2: {
            axis = ceilf(curand_uniform(rand_state) * 3);

            if (comparator(objects[n.s], objects[n.s + 1])) {
                n.bvh->left = objects[n.s];
                n.bvh->right = objects[n.s + 1];
            }
            else {
                n.bvh->left = objects[n.s + 1];
                n.bvh->right = objects[n.s];
            }
        }
        }

        aabb box_left, box_right;

        if (!n.bvh->left->bounding_box(box_left) || !n.bvh->right->bounding_box(box_right)) {
            printf("No bounding box in BVHNode constructor.\n");
        }

        n.bvh->box = surrounding_box(box_left, box_right);
    }
}