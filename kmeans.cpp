#include <iostream>
#include <cmath>
#include <map>
#include <iomanip>
#include "arg_parser.h"

std::vector<Point> points;
std::vector<Point> centroids;
std::map<int, std::vector<int>> final_clusters;

// Returns a float in [0.0, 1.0)
float rand_float() {
    return static_cast<float>(rand()) / static_cast<float>((long long) RAND_MAX + 1);
}

// Initialize k random centroids
void kmeans_init_centroids() {
    for (int i = 0; i < num_clusters; ++i) {
        int idx = (int) (rand_float() * points.size());
        centroids.push_back(points[idx]);
    }
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

void kmeans() {
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

int main(int argc, char **argv) {

    // Parse CLI args, read input file, and set random seed
    parse_args(argc, argv);
    // print_args();

    read_points(points);
    // print_points(points);

    srand(seed);
    kmeans_init_centroids();
    // std::cout << "Before Kmeans:" << std::endl;
    // print_centroids(); 

    kmeans();
    // std::cout << "After Kmeans:" << std::endl;
    print_centroids();
    // print_clusters();

    return 0;
}