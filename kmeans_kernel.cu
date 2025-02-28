#include "kmeans_kernel.cuh"

__global__ void kernel_assign_cluster(float *points, float *centroids, int *assignment, 
                                        int num_points, int num_clusters, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        float min_dist = FLT_MAX;
        int centroid_idx;

        // Find closest centroid for this particular point
        for (int i = 0; i < num_clusters; ++i) {
            float dist = 0.0;
            for (int j = 0; j < dims; ++j) {
                float d = points[idx * dims + j] - centroids[i * dims + j];
                dist += d * d;
            }

            if (dist < min_dist) {
                min_dist = dist;
                centroid_idx = i;
            }
        }

        assignment[idx] = centroid_idx;
    }
}

__global__ void kernel_shmem_assign_cluster(float *points, float *centroids,
                                            int *assignment, int num_points, 
                                            int num_clusters, int dims) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float local_centroids[];

    if (idx < num_points) {

        // For each block, load all centroids into shared memory
        if (threadIdx.x < num_clusters) {
            for (int i = 0; i < dims; ++i) {
                local_centroids[threadIdx.x * dims + i] = centroids[threadIdx.x * dims + i];
            }
        }

        __syncthreads();

        float min_dist = FLT_MAX;
        int centroid_idx;

        // Find closest centroid for this particular point
        for (int i = 0; i < num_clusters; ++i) {
            float dist = 0.0;
            for (int j = 0; j < dims; ++j) {
                float d = points[idx * dims + j] - local_centroids[i * dims + j];
                dist += d * d;
            }

            if (dist < min_dist) {
                min_dist = dist;
                centroid_idx = i;
            }
        }

        assignment[idx] = centroid_idx;
    }
}

__global__ void kernel_compute_new_centroids(float *points, float *centroids, 
                                            int *assignment, int num_points, 
                                            int num_clusters, int dims) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_clusters) {
        int cluster_size = 0;

        // Set centroid position values to 0 before summing and averaging
        for (int i = 0; i < dims; ++i) {
            centroids[idx * dims + i] = 0;
        }

        // Sum
        for (int i = 0; i < num_points; ++i) {

            // Point is assigned to centroid's cluster
            if (assignment[i] == idx) {
                ++cluster_size;

                for (int j = 0; j < dims; ++j) {
                    centroids[idx * dims + j] += points[i * dims + j];
                }
            }
        }

        // Average
        for (int i = 0; i < dims; ++i) {
            centroids[idx * dims + i] /= max(1, cluster_size);
        }
    }
}

__global__ void kernel_check_convergence(float *new_centroids, float *old_centroids, 
                                        bool *converged, int num_clusters, int dims, 
                                        float thresh) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_clusters) {

        // Check convergence criteria for this centroid
        for (int i = 0; i < dims; ++i) {
            if (fabsf(new_centroids[idx * dims + i] - old_centroids[idx * dims + i]) > thresh) {
                converged[idx] = false;
                return;
            }
        }

        converged[idx] = true;
    }
}

__global__ void kernel_shmem_compute_new_centroids(float *points, float *centroids, 
                                                    int *assignment, int *counts, 
                                                    int num_points, int num_clusters, 
                                                    int dims) {

    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared_mem[];
    float *local_centroids = shared_mem;
    int *local_counts = (int *) &shared_mem[num_clusters * dims];

    if (point_idx < num_points) {
        if (threadIdx.x < num_clusters) {
            for (int i = 0; i < dims; ++i) {
                local_centroids[threadIdx.x * dims + i] = 0.0;
            }

            local_counts[threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial centroid sums
        int centroid_idx = assignment[point_idx];
        if (centroid_idx >= 0 && centroid_idx < num_clusters) {
            // printf("Thread %d writing to centroid %d\n", point_idx, centroid_idx);
            atomicAdd(&local_counts[centroid_idx], 1);
            for (int i = 0; i < dims; ++i) {
                atomicAdd(&local_centroids[centroid_idx * dims + i], points[point_idx * dims + i]); 
            }
        }

        __syncthreads();

        // Update global memory with partial sums and counts
        if (threadIdx.x < num_clusters) {
            for (int i = 0; i < dims; ++i) {
                atomicAdd(&centroids[threadIdx.x * dims + i], local_centroids[threadIdx.x * dims + i]);
            }

            atomicAdd(&counts[threadIdx.x], local_counts[threadIdx.x]);
        }
    }
}

__global__ void kernel_shmem_average_centroids(float *centroids, int *counts, int num_clusters, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_clusters) {
        for (int i = 0; i < dims; ++i) {
            centroids[idx * dims + i] /= max(1, counts[idx]);
        }
    }
}

__global__ void kernel_kpp_dist_calc(float *centroids, float *points, int num_clusters, 
                                    int num_points, float *distances, int dims) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        float min_dist = FLT_MAX;
        for (int i = 0; i < num_clusters; ++i) {
            float dist = 0.0;
            for (int j = 0; j < dims; ++j) {
                float d = points[idx * dims + j] - centroids[i * dims + j];
                dist += d * d;
            }

            if (dist < min_dist) {
                min_dist = dist;
            }
        }

        distances[idx] = min_dist;
    }
}