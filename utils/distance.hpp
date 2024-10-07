#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

class ANNS{
    private:
        int vector_dim;
        int k;

        std::vector<std::vector<float>> query_vecs;
        std::vector<std::vector<float>> data_vecs;
        std::vector<std::vector<std::pair<int, double>>> closest_clusters;
        std::vector<std::vector<std::pair<int, double>>> dist_lists{};
    
    public:
        ANNS(const int& dim, const int& k_val, const std::vector<std::vector<float>>& query, const std::vector<std::vector<float>>& data) :
        vector_dim(dim), k(k_val), query_vecs(query), data_vecs(data) {}

        double brute_euclidean(std::vector<float> vec_a, std::vector<float> vec_b){
            double dist = 0.0;
            for(int i = 0; i < vector_dim; ++i){
                dist += std::pow(vec_a[i] - vec_b[i], 2);
            }
            return std::sqrt(dist);
        }

        void sort(std::vector<std::pair<int, double>>& vec) {
            std::sort(vec.begin(), vec.end(),[](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                return a.second < b.second;
            });
        }

        void brute_knn(){
            for(const std::vector<float>& q : query_vecs){
                int i = 0;
                std::vector<std::pair<int, double>> result_q;
                for(std::vector<float>& v : data_vecs){
                    double dist = brute_euclidean(q, v);
                    result_q.push_back(std::make_pair(i, dist));
                    ++i;
                }

                sort(result_q); 

                std::vector<std::pair<int, double>> k_result(result_q.begin(), result_q.begin() + k);
                dist_lists.push_back(k_result);
            }
        }

        void IVF_knn(const std::map<int, std::vector<std::pair<int, std::vector<float>>>>& IVF, int knn_cluster){

            // Find closest clusters to each query
            // Store results in closest_clusters: [[{v_0, d_0}...{v_k, d_k}], [{v_0, d_0}...{v_k, d_k}],...,[{v_0, d_0}...{v_k, d_k}]]  
            //                                               query 1                    query 2         ...          query n
            // v_i: vector index
            // d_i: distance from query to closest vector

            for(const std::vector<float>& q: query_vecs){
                std::vector<std::pair<int, double>> result_q;
                for(const auto& data : IVF){
                    std::vector<float> cluster = data.second[0].second;
                    double dist = brute_euclidean(q, cluster);
                    result_q.push_back(std::make_pair(data.first, dist));
                }
                sort(result_q);
                std::vector<std::pair<int, double>> k_result(result_q.begin(), result_q.begin() + knn_cluster);
                closest_clusters.push_back(k_result);
            }

            // Perform KNN on vectors in each cluster 
            // Loop over each query
            for(int i = 0; i < static_cast<int>(query_vecs.size()); ++i){
                const std::vector<float>& query_vec = query_vecs[i];
                std::vector<std::pair<int, double>> result_q;

                // Loop over each closest cluster to query i
                for(int j = 0; j < static_cast<int>(closest_clusters[i].size()); ++j){
                    int cluster_id = closest_clusters[i][j].first;
                    const std::vector<std::pair<int, std::vector<float>>>& cluster_data = IVF.at(cluster_id);

                    // Loop over each data point in the cluster and calculate distance to query
                    for(int k = 1; k < static_cast<int>(cluster_data.size()); ++k){
                        const std::vector<float>& point_vec = cluster_data[k].second;
                        double dist = brute_euclidean(query_vec, point_vec);

                        int key = cluster_data[k].first;
                        result_q.push_back(std::make_pair(key, dist));
                        ++key;
                    }

                }
                sort(result_q);
        
                std::vector<std::pair<int, double>> k_result(result_q.begin(), result_q.begin() + k);
                dist_lists.push_back(k_result);
            }

        }

        std::vector<std::vector<std::pair<int, double>>> get_dist_lists(){
            return dist_lists;
        }

        std::vector<std::vector<std::pair<int, double>>> get_closest_clusters(){
            return closest_clusters;
        }
};
    