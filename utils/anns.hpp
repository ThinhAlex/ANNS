#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <queue>
#include <utility>
#include "distance.hpp"
#include "pqueue.hpp"

#include <omp.h>
#include <immintrin.h> 

class ANNS{
    private:
        int vector_dim;
        int k; 

        const float* query_vecs;
        const float* data_vecs;

        int query_size;
        int data_size;

        int* dist_lists;
        double runtime;
    
    public:
        ANNS(const int& dim, const int& k_val, const float* query, const float* data, int qsize, int dsize) :
        vector_dim(dim), k(k_val), query_vecs(query), data_vecs(data), query_size(qsize), data_size(dsize) {
            dist_lists = static_cast<int*>(malloc(query_size * k * sizeof(int)));
            runtime = 0.0;
        }

        ~ANNS(){
            free(dist_lists);
        }

        void brute_knn() {
            auto start = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for //schedule(dynamic, 1)
            for (int i = 0; i < query_size; ++i) {
                const float* query_ptr = query_vecs + (i * vector_dim); 
                pqueue_t<int> S(k);
                
                for (int j = 0; j < data_size; ++j) {
                    const float* data_ptr = data_vecs + (j * vector_dim);
                    int dist = compute_distance_squared(vector_dim, query_ptr, data_ptr);
                    S.push(j, dist);
                }

                int* dist_ptr = dist_lists + (i * k);
                for (int s = 0; s < k; ++s) {
                    dist_ptr[s] = S[s];
                }
            }
            auto stop = std::chrono::high_resolution_clock::now();
            runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        }

        void IVF_knn(const float* clusters, std::vector<std::vector<int>> ivf, int num_clusters, int knn_cluster) {     
            auto start = std::chrono::high_resolution_clock::now();  

            #pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < query_size; ++i) {
                const float* query_ptr = query_vecs + (i * vector_dim); 
                pqueue_t<int> C(knn_cluster);

                // Get k closest clusters
                for (int j = 0; j < num_clusters; ++j) {
                    const float* cluster = clusters + j*vector_dim;
                    int dist = compute_distance_squared(vector_dim, query_ptr, cluster);
                    C.push(j, dist);
                }

                pqueue_t<int> S(k);
                // Get closests points from closest clusters    
                for(int s = 0; s < knn_cluster; s++){
                    int cluster_id = C[s];
                    const std::vector<int>& data_list = ivf[cluster_id];
                    for(int id:data_list){
                        const float* point = data_vecs + id*vector_dim;
                        int dist = compute_distance_squared(vector_dim, query_ptr, point);
                        S.push(id, dist);
                    }
                }

                int* dist_ptr = dist_lists + (i * k);
                for (int m = 0; m < k; ++m) {
                    dist_ptr[m] = S[m];
                }
            }
            auto stop = std::chrono::high_resolution_clock::now();
            runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(); 

        }

        int* get_dist_lists(){
            return dist_lists;
        }

        double get_runtime(){
            return runtime;
        }
};
    