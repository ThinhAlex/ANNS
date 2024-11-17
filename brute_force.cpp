#include <iostream>
#include "utils/distance.hpp"
#include "utils/anns.hpp"
#include "utils/data.hpp"
#include "utils/recall.hpp"

#include <omp.h>

int main() {
    GraphData<float> base("data/siftsmall/siftsmall_base.fvecs");
    GraphData<float> query("data/siftsmall/siftsmall_query.fvecs");
    GraphData<int> groundtruth("data/siftsmall/siftsmall_groundtruth.ivecs");

    int base_dim = base.get_vector_dim();
    int gt_dim = groundtruth.get_vector_dim();
    int query_size = query.get_num_vectors();
    int base_size = base.get_num_vectors();  
    int k = 100;

    float* base_data = base.get_data();    
    float* query_data = query.get_data();
    int* gt_data = groundtruth.get_data();

    ANNS ann(base_dim, k, query_data, base_data, query_size, base_size);
    
    // Run brute force k-NN on dataset
    ann.brute_knn();

    // Get runtime
    double run_time = ann.get_runtime();
    std::cout << query_size << std::endl;

    // Calculate throughput and latency
    double throughput = double(query_size) * 1000.0 / run_time;
    double latency = run_time / double(query_size);

    // Get lists of k distances
    int* results = ann.get_dist_lists();  // results is a pointer to the top-k distances

    // Calculate recall
    Recall recall(gt_data, base_data, query_data, results, base_dim, query_size, gt_dim, k);  // Adjust constructor to fit pointer-based data
    double recall_val = recall.get_recall();
    
    int num_threads = 0;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    std::cout << "OpenMP ANN search (" << num_threads << " threads)\n";
    std::cout << "Runtime: " << run_time << " ms\n";
    std::cout << "Throughput: " << throughput << " queries/s\n";
    std::cout << "Latency: " << latency << " ms/query\n";
    std::cout << "Recall: " << recall_val << "\n";

    /*
    // Print the results
    for (int i = 0; i < query_size; ++i) {
        std::cout << "Query vector " << i + 1 << ":\n";
        for (int j = 0; j < k; ++j) {
            std::cout << "Neighbor " << j + 1 << ": (" << results[i * k + j] << ")\n";
        }
        std::cout << std::endl;
    }
    */
    
    return 0;
}
