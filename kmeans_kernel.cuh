#ifndef KMEANS_KERNEL_H
#define KMEANS_KERNEL_H

#include <curand_kernel.h>
#include <cfloat>

__global__ void kernel_assign_cluster(float *points, float *centroids, int *assignment, int num_points, int num_clusters, int dims);
__global__ void kernel_compute_new_centroids(float *points, float *centroids, int *assignment, int num_points, int num_clusters, int dims);
__global__ void kernel_check_convergence(float *new_centroids, float *old_centroids, bool *converged, int num_clusters, int dims, float thresh);

#endif