#include <iostream>
#include "utils/distance.hpp"
#include "utils/anns.hpp"
#include "utils/data.hpp"
#include "utils/kmeans.hpp"
#include "utils/recall.hpp"

#include <omp.h>

int main(){
    GraphData<float> base("data/siftsmall/siftsmall_base.fvecs");
    GraphData<float> query("data/siftsmall/siftsmall_query.fvecs");
    GraphData<int> groundtruth("data/siftsmall/siftsmall_groundtruth.ivecs");

    int base_dim = base.get_vector_dim();
    int gt_dim = groundtruth.get_vector_dim();
    int query_size = query.get_num_vectors();
    int base_size = base.get_num_vectors();  

    float* base_data = base.get_data();    
    float* query_data = query.get_data();
    int* gt_data = groundtruth.get_data();

    int k = 100;
    int num_clusters = 20;
    int knn_cluster = 2; // should be 10% - 25% of num_clusters

    // Run kmeans and get clusters data
    KMeans kmeans(num_clusters, base_dim, base_data, base_size);    
    std::vector<std::vector<int>> ivf = kmeans.build_index();
    const float* clusters = kmeans.get_clusters();
    auto build_time = kmeans.get_build_time();

    
    // Run ivf search
    ANNS ann(base_dim, k, query_data, base_data, query_size, base_size);  
    ann.IVF_knn(clusters, ivf, num_clusters, knn_cluster);
    int* dist_list = ann.get_dist_lists();
    auto search_time = ann.get_runtime();


    // Throughput and latency
    auto throughput = query_size * 1000 / search_time;
    auto latency = search_time / query_size;

    
    //Calculate recall
    Recall recall(gt_data, base_data, query_data, dist_list, base_dim, query_size, gt_dim, k);
    double recall_val = recall.get_recall();

    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }         
    std::cout << "OpenMP ANN search (" << num_threads << " threads)\n";

    std::cout << "Build time: " << build_time << "ms" << std::endl;
    std::cout << "Search time: " << search_time << "ms" << std::endl;
    std::cout << "Throughput: " << throughput << " query/s" << std::endl;
    std::cout << "Latency: " << latency << " ms/query" << std::endl;
    std::cout << "Recall: " << recall_val << std::endl;

    /*
    // Print the results
    for(int i = 0; i < query_size; ++i){
        std::cout << "Query vector " << i+1 << ":" << std::endl;
        for(int j = 0; j < k_search; ++j){
            std::cout << "Neighbor " << j+1 << ": (" << dist_list[i*k_search + j].first << ", " << dist_list[i*k_search + j].second << ")" << std::endl;
        }
        std::cout << std::endl;
    }
    */

}