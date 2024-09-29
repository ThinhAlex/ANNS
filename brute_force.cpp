#pragma once

#include <iostream>
#include <chrono>
#include "distance.hpp"
#include "data.hpp"

int main(){
    GraphData base("data/siftsmall/siftsmall_base.fvecs");
    GraphData query("data/siftsmall/siftsmall_query.fvecs");

    int base_dim = base.get_vector_dim();
    int query_size = query.get_num_vectors();
    int k = 10; 

    std::vector<std::vector<float>> base_data = base.get_data();
    std::vector<std::vector<float>> query_data = query.get_data();
    ANNS ann(base_dim, k, query_data, base_data);

    // Run brute force on dataset
    auto start = std::chrono::high_resolution_clock::now();
    ann.brute_knn();
    auto stop = std::chrono::high_resolution_clock::now();

    auto run_time = std::chrono::duration_cast<std::chrono::seconds>(stop-start);

    // Throughput and latency
    auto throughput = double(query_size) / run_time.count();
    auto latency = run_time.count() * 1000 / double(query_size);

    // Get lists of k distances sorted in ascending order 
    std::vector<std::vector<std::pair<int, double>>> results = ann.get_dist_lists();

    // Print the results
    for(int i = 0; i < results.size(); ++i){
        std::cout << "Query vector " << i+1 << ":" << std::endl;
        for(int j = 0; j < results[i].size(); ++j){
            std::cout << "Neighbor " << j+1 << ": (" << results[i][j].first << ", " << results[i][j].second << ")" << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Runtime: " << run_time.count() << "s" << std::endl;
    std::cout << "Throughput: " << throughput << " query/s" << std::endl;
    std::cout << "Latency: " << latency << " ms/query" << std::endl;

}