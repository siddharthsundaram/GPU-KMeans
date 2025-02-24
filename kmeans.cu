#include <iostream>
#include <cmath>
#include <map>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include "arg_parser.h"
#include "kmeans_kernel.cuh"

std::vector<Point> points;
std::vector<Point> centroids;
std::map<int, std::vector<int>> final_clusters;

#define CHECK_KERNEL(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} \

// Returns a float in [0.0, 1.0)
float rand_float() {
    return static_cast<float>(rand()) / static_cast<float>((long long) RAND_MAX + 1);
}

// Initialize k random centroids
void seq_kmeans_init_centroids() {
    for (int i = 0; i < num_clusters; ++i) {
        int idx = (int) (rand_float() * points.size());
        centroids.push_back(points[idx]);
    }
}

void par_kmeans_init_centroids() {

    // Malloc memory on device
    int *d_rand_idx;
    CHECK_KERNEL(cudaMalloc((int **) &d_rand_idx, num_clusters * sizeof(int)));

    // No mem transfer needed, set dimensions
    dim3 block (num_clusters);
    dim3 grid ((num_clusters + block.x - 1) / block.x);

    // Invoke kernel
    // CHECK_KERNEL(kernel_kmeans_init_centroids<<<grid, block>>>(d_rand_idx, points.size(), seed));

    // Transfer results back to host
    int *h_rand_idx = new int[num_clusters];
    CHECK_KERNEL(cudaMemcpy(h_rand_idx, d_rand_idx, num_clusters * sizeof(int), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < num_clusters; ++i) {
    //     printf(" %d", h_rand_idx[i]);
    // }

    for (int i = 0; i < num_clusters; ++i) {
        centroids.push_back(points[h_rand_idx[i]]);
    }

    CHECK_KERNEL(cudaFree(d_rand_idx));
}

// Calculate Euclidean distance between two points
float dist(Point &p1, Point &p2) {
    float res = 0.0;

    for (int i = 0; i < dims; ++i) {
        res += std::pow(p1.pos[i] - p2.pos[i], 2);
    }

    return std::sqrt(res);
}

std::vector<Point> copy_centroids(std::vector<Point> centroids_to_copy) {
    std::vector<Point> res;

    for (int i = 0; i < num_clusters; ++i) {
        Point centroid = centroids_to_copy[i];
        std::vector<float> new_pos = centroid.pos;

        // Label of -1 indicates that it might not be one of the original points (certainly a centroid though)
        Point centroid_copy = {-1, new_pos};
        res.push_back(centroid_copy);
    }

    return res;
}

// Compute new centroids by taking mean position of all points in the cluster
std::vector<Point> compute_new_centroids() {
    std::vector<Point> new_centroids(num_clusters);

    for (auto it = final_clusters.begin(); it != final_clusters.end(); ++it) {
        int cluster_idx = it->first;
        std::vector<int> point_indices = final_clusters[cluster_idx];
        std::vector<float> new_pos(dims);

        // Compute new centroid position
        for (int i = 0; i < dims; i++) {
            float dim_total = 0.0;
            for (int point_idx : point_indices) {
                Point p = points[point_idx];
                dim_total += p.pos[i];
            }

            new_pos[i] = dim_total / (float) point_indices.size();
        }

        // Store new centroid
        Point new_centroid = {-1, new_pos};
        new_centroids[cluster_idx] = new_centroid;
    }

    return new_centroids;
}

std::vector<Point> default_centroids() {
    std::vector<Point> res;
    for (int i = 0; i < num_clusters; ++i) {
        std::vector<float> pos(dims);
        Point centroid = {-1, pos};
        res.push_back(centroid);
    }

    return res;
}

// Kmeans converges when centroids haven't changed since last iteration
bool converged(std::vector<Point> old_centroids, std::vector<Point> new_centroids) {
    for (int i = 0; i < num_clusters; ++i) {
        Point c1 = old_centroids[i];
        Point c2 = new_centroids[i];

        for (int j = 0; j < dims; ++j) {
            if (std::fabs(c1.pos[j] - c2.pos[j]) > thresh) {
                return false;
            }
        }
    }

    return true;
}

// Perform the Kmeans algorithm
void seq_kmeans() {
    std::cout << "WE DOING SEQUENTIAL KMEANS :(" << std::endl;
    std::vector<Point> old_centroids = default_centroids();
    std::vector<Point> new_centroids = centroids;
    int num_iters = 0;

    while (num_iters++ < max_iter && !converged(old_centroids, new_centroids)) {

        // Store current centroids for convergence criterion
        old_centroids = copy_centroids(new_centroids);

        // Map from centroid/cluster ID to vector of point IDs in that cluster
        final_clusters.clear();

        for (int i = 0; i < points.size(); ++i) {
            Point point = points[i];
            float min_dist = FLT_MAX;
            int centroid_idx;

            // Find closest centroid to point
            for (int j = 0; j < num_clusters; ++j) {
                Point centroid = old_centroids[j];
                float d = dist(point, centroid);

                if (d < min_dist) {
                    min_dist = d;
                    centroid_idx = j;
                }
            }

            // Add point to cluster
            final_clusters[centroid_idx].push_back(i);
        }

        new_centroids = compute_new_centroids();
    }

    centroids = new_centroids;
}

void par_kmeans() {

    std::cout << "WE DOING PARALLEL KMEANS BABY" << std::endl;

    // Set up for CUDA
    float *h_points = new float[points.size() * dims];
    for (int i = 0; i < points.size(); ++i) {
        Point p = points[i];
        for (int j = 0; j < dims; ++j) {
            h_points[i * dims + j] = p.pos[j];
        }
    }

    float *h_centroids = new float[num_clusters * dims];
    for (int i = 0; i < num_clusters; ++i) {
        Point c = centroids[i];
        for (int j = 0; j < dims; ++j) {
            h_centroids[i * dims + j] = c.pos[j];
        }
    }

    float *h_old_centroids = new float[num_clusters * dims];
    bool *h_converged = new bool[num_clusters];
    int num_iters = 0;
    bool converged = false;
    float *d_points, *d_centroids, *d_old_centroids;
    int *d_assignment;
    bool *d_converged;
    CHECK_KERNEL(cudaMalloc(&d_points, points.size() * dims * sizeof(float)));
    CHECK_KERNEL(cudaMalloc(&d_centroids, num_clusters * dims * sizeof(float)));
    CHECK_KERNEL(cudaMalloc(&d_old_centroids, num_clusters * dims * sizeof(float)));
    CHECK_KERNEL(cudaMalloc(&d_assignment, points.size() * sizeof(int)));
    CHECK_KERNEL(cudaMalloc(&d_converged, num_clusters * sizeof(bool)));
    CHECK_KERNEL(cudaMemcpy(d_points, h_points, points.size() * dims * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_KERNEL(cudaMemcpy(d_centroids, h_centroids, num_clusters * dims * sizeof(float), cudaMemcpyHostToDevice));
    // CHECK_KERNEL(cudaMemcpy(d_old_centroids, h_old_centroids, num_clusters * dims * sizeof(float)));

    // Saw that RTX 6000 can only have 1024 threads per block, check this
    int num_point_threads = std::min((int) points.size(), 1024);
    int num_cluster_threads = std::min(num_clusters, 1024);
    dim3 point_block (num_point_threads);
    dim3 point_grid (points.size() + point_block.x - 1 / point_block.x);
    dim3 centroid_block (num_cluster_threads);
    dim3 centroid_grid (num_clusters + centroid_block.x - 1 / centroid_block.x);

    while (num_iters++ < max_iter && !converged) {
        // std::memcpy(h_old_centroids, h_centroids, num_clusters * dims * sizeof(float));

        // Parallelize cluster assignment
        kernel_assign_cluster<<<point_grid, point_block>>>(d_points, d_centroids, d_assignment, points.size(), num_clusters, dims);

        // Parallelize new centroid computation
        kernel_compute_new_centroids<<<centroid_grid, centroid_block>>>(d_points, d_centroids, d_assignment, points.size(), num_clusters, dims);

        // Parallelize convergence check
        kernel_check_convergence<<<centroid_grid, centroid_block>>>(d_centroids, d_old_centroids, d_converged, num_clusters, dims, thresh);

        // Set up for next iteration
        CHECK_KERNEL(cudaMemcpy(h_converged, d_converged, num_clusters * sizeof(bool), cudaMemcpyDeviceToHost));
        bool local_converged = true;
        for (int i = 0; i < num_clusters; ++i) {
            if (!h_converged[i]) {
                local_converged = false;
            }
        }

        converged = local_converged;
        
        CHECK_KERNEL(cudaMemcpy(d_old_centroids, d_centroids, num_clusters * dims * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Copy centroid data back to host and clean up
    CHECK_KERNEL(cudaMemcpy(h_centroids, d_centroids, num_clusters * dims * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_clusters; ++i) {
        std::vector<float> pos(dims);
        for (int j = 0; j < dims; ++j) {
            pos[j] = h_centroids[i * dims + j];
        }

        Point c = {-1, pos};
        centroids[i] = c;
    }

    std::cout << "WE FINISHED PARALLEL KMEANS BABY" << std::endl;
}

void print_centroids() {
    // print_points(centroids);

    for (int i = 0; i < num_clusters; ++i) {
        Point centroid = centroids[i];
        std::cout << i << " ";

        for (int j = 0; j < dims; ++ j) {
            std::cout << std::setprecision(5) << centroid.pos[j] << " ";
        }

        std::cout << std::endl;
    }
}

void print_clusters() {
    for (auto it = final_clusters.begin(); it != final_clusters.end(); ++it) {
        int cluster_num = it->first;
        std::vector<int> point_indices = final_clusters[cluster_num];
        std::cout << "CLUSTER " << cluster_num << ": [";
        for (int i = 0; i < point_indices.size(); ++i) {
            std::cout << point_indices[i];
            if (i < point_indices.size() - 1) {
                std::cout << ", ";
            }
        }

        std::cout << "]" << std::endl;
    }
}

void print_output() {
    if (output_centroids) {
        print_centroids();
    } else {
        std::map<int, int> point_labels;
        for (auto it = final_clusters.begin(); it != final_clusters.end(); ++it) {
            int label = it->first;
            std::vector<int> cluster_points = final_clusters[label];

            for (int i = 0; i < cluster_points.size(); ++i) {
                point_labels[cluster_points[i]] = label;
            }
        }

        for (int i = 0; i < points.size(); ++i) {
            printf(" %d", point_labels[i]);
        }

        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {

    // Parse CLI args, read input file, and set random seed
    parse_args(argc, argv);
    // print_args();

    read_points(points);
    // print_points(points);

    srand(seed);
    seq_kmeans_init_centroids();

    // Sequential implementation
    if (!use_gpu) {

        // std::cout << "Before Kmeans:" << std::endl;
        // print_centroids(); 

        seq_kmeans();
        // std::cout << "After Kmeans:" << std::endl;
        print_output();
        // print_clusters();

    // Parallel CUDA implementation
    } else {
        CHECK_KERNEL(cudaSetDevice(0));
        par_kmeans();
        print_output();
    }

    return 0;
}