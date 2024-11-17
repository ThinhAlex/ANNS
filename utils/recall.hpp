#pragma once
#include <omp.h>
#include <iostream>
#include "distance.hpp"

class Recall {
private:
    const int* gt;                        
    const float* base;                    
    const float* query;                   
    const int* knn_results; 
    int vector_dim;                       // Dimension of each vector
    int num_query;                        // Number of query vectors
    int gt_k;                             // Number of ground truth neighbors per query
    int query_k;                          // Number of predicted neighbors per query
    double recall;

public:
    Recall(const int* gt_data, const float* base_data, const float* query_data, 
           const int* knn_results_data, int vector_dim, int num_query, int gt_k, int query_k)
        : gt(gt_data), base(base_data), query(query_data), knn_results(knn_results_data),
          vector_dim(vector_dim), num_query(num_query), gt_k(gt_k), query_k(query_k), recall(0.0) 
    {
        if (query_k <= gt_k) {
            #pragma omp parallel for reduction(+:recall)
            for (int i = 0; i < num_query; ++i) {
                int correct_count = 0;
                for (int j = 0; j < query_k; ++j) {
                    int predict_id = knn_results[i * query_k + j];
                    int gt_id = gt[i * gt_k + j];
                    if (predict_id == gt_id) {
                        ++correct_count;
                    } else {
                        const float* predict_vec = base + predict_id * vector_dim;
                        const float* gt_vec = base + gt_id * vector_dim;
                        const float* query_vec = query + i * vector_dim;

                        int dist_predict = compute_distance_squared(vector_dim, predict_vec, query_vec);
                        int dist_gt =  compute_distance_squared(vector_dim, gt_vec, query_vec);
                        if (dist_predict == dist_gt) {
                            ++correct_count;
                        }
                    }
                }
                recall += static_cast<double>(correct_count) / query_k;
            }
            recall /= num_query;
        } else {
            #pragma omp parallel for reduction(+:recall)
            for (int i = 0; i < num_query; ++i) {
                int correct_count = 0;
                for (int j = 0; j < gt_k; ++j) {
                    int predict_id = knn_results[i * query_k + j];
                    int gt_id = gt[i * gt_k + j];
                    if (predict_id == gt_id) {
                        ++correct_count;
                    } else {
                        const float* predict_vec = base + predict_id * vector_dim;
                        const float* gt_vec = base + gt_id * vector_dim;
                        const float* query_vec = query + i * vector_dim;

                        int dist_predict = compute_distance_squared(vector_dim, predict_vec, query_vec);
                        int dist_gt = compute_distance_squared(vector_dim, gt_vec, query_vec);
                        if (dist_predict == dist_gt) {
                            ++correct_count;
                        }
                    }
                }
                recall += static_cast<double>(correct_count) / gt_k;
            }
            recall /= num_query;
        }
    }

    double get_recall() const {
        return recall;
    }
};
