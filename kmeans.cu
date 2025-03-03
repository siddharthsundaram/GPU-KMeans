#include <iostream>
#include <cmath>
#include <map>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <chrono>
#include "arg_parser.h"
#include "kmeans_kernel.cuh"

std::vector<Point> points;
std::vector<Point> centroids;
int *final_clusters;
int num_iters = 0;
float total_data_transfer_time = 0.0;
cudaEvent_t start, stop;

// Inspired by Ed post and textbook
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
    // std::cout << "SEQ INIT" << std::endl;
    for (int i = 0; i < num_clusters; ++i) {
        int idx = (int) (rand_float() * points.size());
        // std::cout << "RANDOM IDX IS " << idx << std::endl;
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
    std::vector<int> counts(num_clusters);
    std::vector<std::vector<float>> centroid_positions(num_clusters, std::vector<float>(dims));

    for (int i = 0; i < points.size(); ++i) {
        int cluster_idx = final_clusters[i];
        counts[cluster_idx] += 1;
        for (int j = 0; j < dims; ++j) {
            centroid_positions[cluster_idx][j] += points[i].pos[j];
        }
    }

    for (int i = 0; i < num_clusters; ++i) {
        for (int j = 0; j < dims; ++j) {
            centroid_positions[i][j] /= max(1, counts[i]);
        }

        Point new_centroid = {-1, centroid_positions[i]};
        new_centroids[i] = new_centroid;
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
    // std::cout << "WE DOING SEQUENTIAL KMEANS :(" << std::endl;
    static std::vector<Point> old_centroids = default_centroids();
    static std::vector<Point> new_centroids = centroids;
    final_clusters = new int[points.size()];

    while (num_iters < max_iter && !converged(old_centroids, new_centroids)) {

        ++num_iters;

        // Store current centroids for convergence criterion
        old_centroids = new_centroids;

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
            final_clusters[i] = centroid_idx;
        }

        new_centroids = compute_new_centroids();
    }

    centroids = new_centroids;
}

void par_kmeans() {

    // std::cout << "WE DOING PARALLEL KMEANS BABY" << std::endl;

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

    final_clusters = new int[points.size()];
    float *h_old_centroids = new float[num_clusters * dims];
    bool *h_converged = new bool[num_clusters];
    bool converged = false;
    float *d_points, *d_centroids, *d_old_centroids;
    int *d_assignment;
    bool *d_converged;
    float data_transfer_time;

    CHECK_KERNEL(cudaMalloc(&d_points, points.size() * dims * sizeof(float)));
    CHECK_KERNEL(cudaMalloc(&d_centroids, num_clusters * dims * sizeof(float)));
    CHECK_KERNEL(cudaMalloc(&d_old_centroids, num_clusters * dims * sizeof(float)));
    CHECK_KERNEL(cudaMalloc(&d_assignment, points.size() * sizeof(int)));
    CHECK_KERNEL(cudaMalloc(&d_converged, num_clusters * sizeof(bool)));

    CHECK_KERNEL(cudaEventRecord(start, 0));
    CHECK_KERNEL(cudaMemcpy(d_points, h_points, points.size() * dims * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_KERNEL(cudaMemcpy(d_centroids, h_centroids, num_clusters * dims * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_KERNEL(cudaEventRecord(stop, 0));
    CHECK_KERNEL(cudaEventSynchronize(stop));

    CHECK_KERNEL(cudaEventElapsedTime(&data_transfer_time, start, stop));
    total_data_transfer_time += data_transfer_time;

    // Saw that RTX 6000 can only have 1024 threads per block, check this
    int num_point_threads = std::min((int) points.size(), threads_per_block);
    int num_cluster_threads = std::min(num_clusters, threads_per_block);
    dim3 point_block (num_point_threads);

    dim3 point_grid ((points.size() + point_block.x - 1) / point_block.x);
    dim3 centroid_block (num_cluster_threads);
    dim3 centroid_grid ((num_clusters + centroid_block.x - 1) / centroid_block.x);
    int assignment_shared_mem_size = sizeof(float) * dims * (num_clusters);
    int compute_shared_mem_size = sizeof(float) * num_clusters * dims + sizeof(int) * num_clusters;

    while (num_iters < max_iter && !converged) {

        ++num_iters;
        
        if (use_shared_mem) {
            // std::cout << "WE USING SHARED MEMORY BABY" << std::endl;

            // Parallelize cluster assignment
            kernel_shmem_assign_cluster<<<point_grid, point_block, assignment_shared_mem_size>>>(d_points, d_centroids, d_assignment, points.size(), num_clusters, dims);

            // Parallelize new centroid computation
            int *d_counts;
            CHECK_KERNEL(cudaMalloc(&d_counts, sizeof(int) * num_clusters));
            cudaMemset(d_centroids, 0, num_clusters * dims * sizeof(float));
            cudaMemset(d_counts, 0, num_clusters * sizeof(int));
            kernel_shmem_compute_new_centroids<<<point_grid, point_block, compute_shared_mem_size>>>(d_points, d_centroids, d_assignment, d_counts, points.size(), num_clusters, dims);
            cudaDeviceSynchronize();
            kernel_shmem_average_centroids<<<centroid_grid, centroid_block>>>(d_centroids, d_counts, num_clusters, dims);
            CHECK_KERNEL(cudaFree(d_counts));
        } else {

            // Parallelize cluster assignment
            kernel_assign_cluster<<<point_grid, point_block>>>(d_points, d_centroids, d_assignment, points.size(), num_clusters, dims);

            // Parallelize new centroid computation
            kernel_compute_new_centroids<<<centroid_grid, centroid_block>>>(d_points, d_centroids, d_assignment, points.size(), num_clusters, dims);
        }

        // Parallelize convergence check
        kernel_check_convergence<<<centroid_grid, centroid_block>>>(d_centroids, d_old_centroids, d_converged, num_clusters, dims, thresh);

        // Set up for next iteration
        float time_i;
        CHECK_KERNEL(cudaEventRecord(start, 0));
        CHECK_KERNEL(cudaMemcpy(h_converged, d_converged, num_clusters * sizeof(bool), cudaMemcpyDeviceToHost));
        CHECK_KERNEL(cudaMemcpy(d_old_centroids, d_centroids, num_clusters * dims * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_KERNEL(cudaEventRecord(stop, 0));
        CHECK_KERNEL(cudaEventSynchronize(stop));
        CHECK_KERNEL(cudaEventElapsedTime(&time_i, start, stop));
        total_data_transfer_time += time_i;

        bool local_converged = true;
        for (int i = 0; i < num_clusters; ++i) {
            if (!h_converged[i]) {
                local_converged = false;
            }
        }

        converged = local_converged;
        
    }

    // Copy centroid data back to host and clean up
    float time_e;
    CHECK_KERNEL(cudaEventRecord(start, 0));
    CHECK_KERNEL(cudaMemcpy(h_centroids, d_centroids, num_clusters * dims * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_KERNEL(cudaMemcpy(final_clusters, d_assignment, points.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_KERNEL(cudaEventRecord(stop, 0));
    CHECK_KERNEL(cudaEventSynchronize(stop));
    CHECK_KERNEL(cudaEventElapsedTime(&time_e, start, stop));
    total_data_transfer_time += time_e;

    CHECK_KERNEL(cudaDeviceSynchronize());
    CHECK_KERNEL(cudaFree(d_points));
    CHECK_KERNEL(cudaFree(d_centroids));
    CHECK_KERNEL(cudaFree(d_old_centroids));
    CHECK_KERNEL(cudaFree(d_assignment));
    CHECK_KERNEL(cudaFree(d_converged));

    for (int i = 0; i < num_clusters; ++i) {
        std::vector<float> pos(dims);
        for (int j = 0; j < dims; ++j) {
            pos[j] = h_centroids[i * dims + j];
        }

        Point c = {-1, pos};
        centroids[i] = c;
    }

    // std::cout << "WE FINISHED PARALLEL KMEANS BABY" << std::endl;
}

void print_centroids() {
    for (int i = 0; i < num_clusters; ++i) {
        Point centroid = centroids[i];
        std::cout << i << " ";

        for (int j = 0; j < dims; ++ j) {
            std::cout << std::setprecision(5) << centroid.pos[j] << " ";
        }

        std::cout << std::endl;
    }
}

void print_output() {
    if (output_centroids) {
        print_centroids();
    } else {
        printf("clusters:");
        for (int i = 0; i < points.size(); ++i) {
            printf(" %d", final_clusters[i]);
        }

        std::cout << std::endl;
    }
}

void kmeanspp_init_centroids() {
    // std::cout << "WE DOING KMEANS++ BABY" << std::endl;
    int first_centroid_idx = (int) (rand_float() * points.size());
    // std::cout << first_centroid_idx << std::endl;
    Point first_centroid = points[first_centroid_idx];
    centroids.push_back(first_centroid);

    // Kernel setup
    float *h_points = new float[points.size() * dims];
    for (int i = 0; i < points.size(); ++i) {
        Point p = points[i];
        for (int j = 0; j < dims; ++j) {
            h_points[i * dims + j] = p.pos[j];
        }
    }

    float *h_centroids = new float[num_clusters * dims];
    for (int i = 0; i < dims; ++i) {
        h_centroids[i] = first_centroid.pos[i];
    }

    float *h_distances = new float[points.size()];

    float *d_points, *d_centroids, *d_distances;
    CHECK_KERNEL(cudaMalloc(&d_points, points.size() * dims * sizeof(float)));
    CHECK_KERNEL(cudaMalloc(&d_centroids, num_clusters * dims * sizeof(float)));
    CHECK_KERNEL(cudaMalloc(&d_distances, points.size() * sizeof(float)));

    float transfer_time;
    CHECK_KERNEL(cudaEventRecord(start, 0));
    CHECK_KERNEL(cudaMemcpy(d_points, h_points, points.size() * dims * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_KERNEL(cudaEventRecord(stop, 0));
    CHECK_KERNEL(cudaEventSynchronize(stop));
    CHECK_KERNEL(cudaEventElapsedTime(&transfer_time, start, stop));
    total_data_transfer_time += transfer_time;

    int num_point_threads = std::min((int) points.size(), threads_per_block);
    dim3 point_block (num_point_threads);
    dim3 point_grid ((points.size() + point_block.x - 1) / point_block.x);

    while (centroids.size() < num_clusters) {

        // TODO: If time permits, implement shared memory for kpp
        if (use_gpu) {

            // Add new centroid to device memory
            float time1, time2;
            CHECK_KERNEL(cudaEventRecord(start, 0));
            CHECK_KERNEL(cudaMemcpy(&d_centroids[(centroids.size() - 1) * dims], 
                                    &h_centroids[(centroids.size() - 1) * dims], 
                                    dims * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_KERNEL(cudaEventRecord(stop, 0));
            CHECK_KERNEL(cudaEventSynchronize(stop));
            CHECK_KERNEL(cudaEventElapsedTime(&time1, start, stop));
            total_data_transfer_time += time1;
            
            kernel_kpp_dist_calc<<<point_grid, point_block>>>(d_centroids, d_points, centroids.size(), points.size(), d_distances, dims);

            CHECK_KERNEL(cudaEventRecord(start, 0));
            CHECK_KERNEL(cudaMemcpy(h_distances, d_distances, points.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_KERNEL(cudaEventRecord(stop, 0));
            CHECK_KERNEL(cudaEventSynchronize(stop));
            CHECK_KERNEL(cudaEventElapsedTime(&time2, start, stop));
            total_data_transfer_time += time2;
        } else {
            for (int i = 0; i < points.size(); ++i) {
                Point point = points[i];
                float min_dist = FLT_MAX;

                // Find closest centroid to point
                for (int j = 0; j < centroids.size(); ++j) {
                    Point centroid = centroids[j];
                    float d = dist(point, centroid);

                    if (d < min_dist) {
                        min_dist = d;
                    }
                }

                h_distances[i] = min_dist;
            }
        }

        // Determine next centroid
        float total_dist = 0.0;
        for (int i = 0; i < points.size(); ++i) {
            total_dist += h_distances[i];
        }

        float target = rand_float() * total_dist;
        float dist = 0.0;
        // std::cout << "DISTANCES: " << std::endl;
        for (int i = 0; i < points.size(); ++i) {
            dist += h_distances[i];
            // std::cout << h_distances[i] << std::endl;
            if (target < dist) {
                // std::cout << i << std::endl;
                Point new_centroid = points[i];

                for(int j = 0; j < dims; ++j) {
                    h_centroids[centroids.size() * dims + j] = new_centroid.pos[j];
                }

                centroids.push_back(new_centroid);
                break;
            }
        }
    }

    CHECK_KERNEL(cudaFree(d_points));
    CHECK_KERNEL(cudaFree(d_centroids));
    CHECK_KERNEL(cudaFree(d_distances));
}

int main(int argc, char **argv) {

    // cudaDeviceProp prop;
    // int device_id = 0;

    // cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    // if (err != cudaSuccess) {
    //     std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
    //     return 1;
    // }

    // std::cout << "Device Name: " << prop.name << std::endl;
    // std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    // std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    // std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    // std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    // std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    // std::cout << "Max Threads Dim: (" << prop.maxThreadsDim[0] << ", "
    //           << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    // std::cout << "Max Grid Size: (" << prop.maxGridSize[0] << ", "
    //           << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    // std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    // std::cout << "Warp Size is: " << prop.warpSize << std::endl;

    CHECK_KERNEL(cudaEventCreate(&start));
    CHECK_KERNEL(cudaEventCreate(&stop));
    auto start = std::chrono::high_resolution_clock::now();

    // Parse CLI args, read input file, and set random seed
    parse_args(argc, argv);
    read_points(points);
    srand(seed);

    // Kmeans++ implementation
    if (use_kpp) {
        kmeanspp_init_centroids();

    // Normal kmeans initialization
    } else {
        seq_kmeans_init_centroids();
    }

    // Sequential implementation
    if (!use_gpu && !use_shared_mem && !use_kpp) {
        seq_kmeans();

    // Parallel CUDA implementation
    } else {
        CHECK_KERNEL(cudaSetDevice(0));

        par_kmeans();
    }

    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    // printf("%d,%lf\n", num_iters, duration.count() / num_iters);
    // print_output();
    // std::cout << "TOTAL DATA TRANSFER TIME: " << total_data_transfer_time << " ms" << std::endl;
    // std::cout << "TOTAL TIME: " << duration.count() << " ms" << std::endl;
    printf("%f,%lf\n", total_data_transfer_time, duration.count());

    return 0;
}