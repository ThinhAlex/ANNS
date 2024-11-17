#pragma once

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include <limits>
#include <algorithm>
#include <vector>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include "distance.hpp"
#include <omp.h>


class KMeans {
private:
    int num_clusters;
    int base_dim;
    float* base_data;
    float* clusters;
    int* assignments; 
    int num_points;

    double build_time;

public:
    KMeans(int k, int dim, float* data, int data_size)
        : num_clusters(k), base_dim(dim), base_data(data), num_points(data_size) {
        
        clusters = static_cast<float*>(aligned_alloc(32, num_clusters * base_dim * sizeof(float)));
        if (!clusters) throw std::bad_alloc();

        assignments = static_cast<int*>(malloc(num_points * sizeof(int)));
        if (!assignments) throw std::bad_alloc();
    }

    ~KMeans() {
        free(clusters);
        free(assignments);
    }

    void initialize_clusters() {
        std::mt19937 gen(42);
        std::uniform_int_distribution<> distr(0,500);

        #pragma omp parallel for
        for (int i = 0; i < num_clusters; ++i) {
            for (int j = 0; j < base_dim; ++j) {
                clusters[i * base_dim + j] = static_cast<float>(distr(gen)); 
            }
        }
    }

    void assign_clusters() {
        for (int i = 0; i < num_points; ++i) {
            float* point = base_data + i * base_dim;
            int nearest_cluster = -1;
            float min_dist = std::numeric_limits<float>::max();
            for (int j = 0; j < num_clusters; ++j) {
                float* cluster = clusters + j * base_dim;
                float dist = compute_distance_squared(base_dim, point, cluster);

                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_cluster = j;
                }
            }            
            assignments[i] = nearest_cluster;
        }
    }

    void update_clusters() {
        // Temporary storage for cluster centers and point counts
        float* new_clusters = static_cast<float*>(calloc(num_clusters * base_dim, sizeof(float)));
        int* counts = static_cast<int*>(calloc(num_clusters, sizeof(int)));

        for (int i = 0; i < num_points; ++i) {
            int cluster_id = assignments[i];
            float* point = base_data + i * base_dim;
            float* cluster = new_clusters + cluster_id * base_dim;

            for (int j = 0; j < base_dim; ++j) {
                cluster[j] += point[j];
            }
            counts[cluster_id]++;
        }

        // Compute the new cluster centers
        for (int i = 0; i < num_clusters; ++i) {
            float* cluster = new_clusters + i * base_dim;
            int count = counts[i];
            if (count > 0) {
                for (int j = 0; j < base_dim; ++j) {
                    cluster[j] /= count;
                }
            }
        }

        std::memcpy(clusters, new_clusters, num_clusters * base_dim * sizeof(float));

        free(new_clusters);
        free(counts);
    }

    void run_kmeans(int max_iterations) {
        initialize_clusters();

        #pragma omp parallel for
        for (int iter = 0; iter < max_iterations; ++iter) {
            assign_clusters();
            update_clusters();
        }
    }

    float* get_clusters() const {
        return clusters;
    }

    int* get_assignments() const {
        return assignments;
    }

    std::vector<std::vector<int>> build_index() {
        auto start = std::chrono::high_resolution_clock::now();
        run_kmeans(300);  

        std::vector<std::vector<int>> ivf(num_clusters);

        for(int i = 0; i < num_points; ++i){
            ivf[assignments[i]].push_back(i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return ivf;
    }

    double get_build_time() const {
        return build_time;
    }

    void print_clusters() const {
        std::cout << "Cluster centers:\n";
        for (int i = 0; i < num_clusters; ++i) {
            std::cout << "Cluster " << i << ": ";
            for (int j = 0; j < base_dim; ++j) {
                std::cout << clusters[i * base_dim + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};
