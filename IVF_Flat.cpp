#include <iostream>
#include <chrono>
#include "utils/distance.hpp"
#include "utils/data.hpp"
#include "utils/kmeans.hpp"
#include "utils/recall.hpp"

int main(){
    GraphData<float> base("data/siftsmall/siftsmall_base.fvecs");
    GraphData<float> query("data/siftsmall/siftsmall_query.fvecs");
    GraphData<int> groundtruth("data/siftsmall/siftsmall_groundtruth.ivecs");

    int dim = base.get_vector_dim();
    int query_size = query.get_num_vectors();
    std::vector<std::vector<float>> base_data = base.get_data();
    std::vector<std::vector<float>> query_data = query.get_data();
    std::vector<std::vector<int>> gt_data = groundtruth.get_data();

    /*
    Parameters for K-means clustering:
        k_cluster: 
            number of clusters at each step of bisect kmeans

        k_max: 
            stopping condition of bisect kmeans (2.5% of dataset size)

        k_cluster_search: 
            search for k nearest clusters (10% of k_max)

        k_search: 
            search for k nearest neighbors
    */

    int k_cluster = 16;
    int k_max = base.get_num_vectors()*0.1;
    int k_cluster_search = k_max*0.3;
    int k_search = 10;

    KMeans kmeans(k_cluster, dim, base_data);
    ANNS ann(dim, k_search, query_data, base_data);

    // Run bisect kmeans and build index
    auto start_build = std::chrono::high_resolution_clock::now();
    kmeans.build_index(k_max);
    std::map<int, std::vector<std::pair<int, std::vector<float>>>> IVF = kmeans.get_IVF();
    auto stop_build = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::seconds>(stop_build-start_build);

    // Run IVF search on query data
    auto start_search = std::chrono::high_resolution_clock::now();
    ann.IVF_knn(IVF, k_cluster_search);
    auto stop_search = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::seconds>(stop_search-start_search);

    // Throughput and latency
    auto throughput = double(query_size) / search_time.count();
    auto latency = double(search_time.count()) * 1000 / query_size;

    // Get lists of k distances sorted in ascending order
    std::vector<std::vector<std::pair<int, double>>> results = ann.get_dist_lists();

    //Calculate recall
    Recall recall(gt_data, results);
    double recall_val = recall.get_recall();

    // Print the results
    for(int i = 0; i < results.size(); ++i){
        std::cout << "Query vector " << i+1 << ":" << std::endl;
        for(int j = 0; j < results[i].size(); ++j){
            std::cout << "Neighbor " << j+1 << ": (" << results[i][j].first << ", " << results[i][j].second << ")" << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Build time: " << build_time.count() << "s" << std::endl;
    std::cout << "Index size: " << kmeans.get_IVF_size() << std::endl;

    std::cout << "Search time: " << search_time.count() << "s" << std::endl;
    std::cout << "Throughput: " << throughput << " query/s" << std::endl;
    std::cout << "Latency: " << latency << " ms/query" << std::endl;
    std::cout << "Recall: " << recall_val << std::endl;

}