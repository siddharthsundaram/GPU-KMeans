#include "kmeans_kernel.cuh"

// __global__ void kernel_kmeans_init_centroids(int *indices, int num_points, int seed) {
//     // float rand;
//     // rand_float(&rand);
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     curandState state;
//     curand_init(seed, idx, 0, state);
//     float rand = curand_uniform(&state);


//     indices[idx] = (int) (rand * num_points);
// }

// __device__ void rand_float(float *res) {
//     *res = static_cast<float>(rand()) / static_cast<float>((long long) RAND_MAX + 1);
// }

__global__ void kernel_assign_cluster(float *points, float *centroids, int *assignment, int num_points, int num_clusters, int dims) {
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

__global__ void kernel_compute_new_centroids(float *points, float *centroids, int *assignment, int num_points, int num_clusters, int dims) {
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

__global__ void kernel_check_convergence(float *new_centroids, float *old_centroids, bool *converged, int num_clusters, int dims, float thresh) {
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