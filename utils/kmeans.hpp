#pragma once

#include <map>
#include <vector>
#include <random>
#include <limits>
#include <map>

class KMeans{
    private:
        int num_clusters;
        int base_dim;
        std::vector<std::vector<float>> base_data;
        std::vector<std::vector<std::pair<int, std::vector<float>>>> kmeans_results;
        std::map<int, std::vector<std::pair<int, std::vector<float>>>> IVF;
        

    public:
        KMeans(const int& k, const int& vec_dim, const std::vector<std::vector<float>>& dataset) : 
        num_clusters(k), base_dim(vec_dim), base_data(dataset) {}
        
        double euclidean(std::vector<float> vec_a, std::vector<float> vec_b){
            double dist = 0.0;
            for(int i = 0; i < base_dim; ++i){
                dist += std::pow(vec_a[i] - vec_b[i], 2);
            }
            return std::sqrt(dist);
        }

        std::vector<std::vector<float>> create_clusters(){
            const int min_val = 0;
            const int max_val = 120;

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distr(min_val, max_val); 

            std::vector<std::vector<float>> clusters(num_clusters);
            for(int j = 0; j < num_clusters; ++j){
                std::vector<float> cluster(base_dim);
                for (int i = 0; i < base_dim; ++i) {
                    cluster[i] = distr(gen); 
                }
                clusters[j] = cluster;
            }
            return clusters;
        }

        void assign_clusters_initial(std::vector<std::vector<float>> data, std::vector<std::vector<float>> clusters_list){
            // Assign each data point to the nearest cluster 
            // [[[c1], [p1], ... [pk]], [[c2], [p1], ... [pk]]]
            std::vector<std::vector<std::pair<int, std::vector<float>>>> kmeans_list;

            // Push clusters to kmeans_list
            for(std::vector<float> cluster: clusters_list){
                std::vector<std::pair<int, std::vector<float>>> cluster_i = {{-1, cluster}};
                kmeans_list.push_back(cluster_i);
            }

            // Calculate distance from each cluster to data point
            // Assign each data point to the nearest cluster  
            for(int j = 0; j < data.size(); ++j){
                const std::vector<float>& elem = data[j];
                double min_dist = std::numeric_limits<double>::infinity();
                int min_cluster_id = -1; 
                for(int i = 0; i < num_clusters; ++i){
                    double dist = euclidean(elem, clusters_list[i]);
                    if(dist < min_dist){
                        min_dist = dist;
                        min_cluster_id = i;
                    }
                }
                kmeans_list[min_cluster_id].push_back({j, elem}); 
            }
            kmeans_results.insert(kmeans_results.end(), kmeans_list.begin(), kmeans_list.end());
        }

        void assign_clusters(const std::vector<std::pair<int, std::vector<float>>>& cluster_data, const std::vector<std::vector<float>>& clusters_list){
            // Assign each data point to the nearest cluster 
            // [[[c1], [p1], ... [pk]], [[c2], [p1], ... [pk]]]
            std::vector<std::vector<std::pair<int, std::vector<float>>>> kmeans_list;

            // Push clusters to kmeans_list
            for(std::vector<float> cluster: clusters_list){
                std::vector<std::pair<int, std::vector<float>>> cluster_i = {{-1, cluster}};
                kmeans_list.push_back(cluster_i);
            }

            // Calculate distance from each cluster to data point
            // Assign each data point to the nearest cluster  
            for(int j = 0; j < cluster_data.size(); ++j){
                const std::vector<float>& elem = cluster_data[j].second;
                double min_dist = std::numeric_limits<double>::infinity();
                int min_cluster_id = -1; 
                for(int i = 0; i < num_clusters; ++i){
                    double dist = euclidean(elem, clusters_list[i]);
                    if(dist < min_dist){
                        min_dist = dist;
                        min_cluster_id = i;
                    }
                }
                kmeans_list[min_cluster_id].push_back({cluster_data[j].first, elem}); 
            }
            kmeans_results.insert(kmeans_results.end(), kmeans_list.begin(), kmeans_list.end());
        }

        int find_largest_cluster(){
            int largest_size = 0;
            int largest_cluster_id = -1;
            for(int i = 0; i < static_cast<int>(kmeans_results.size()); ++i){
                if(static_cast<int>(kmeans_results[i].size()) > largest_size){
                    largest_size = static_cast<int>(kmeans_results[i].size());
                    largest_cluster_id = i;
                }
            }
            return largest_cluster_id;
        }

        void bisect_kmeans(const int& max_num_clusters){ 
            // For intitialization, we will run kmeans on the entire data 
            std::vector<std::vector<float>> initial_clusters = create_clusters();
            assign_clusters_initial(base_data, initial_clusters);
            while(true){
                // Check stopping condition
                // kmeans_results: vector<vector<pair<int, vec<float>>>>
                if(static_cast<int>(kmeans_results.size()) >= max_num_clusters){
                    break;
                }else{
                    // Find the largest cluster
                    int largest_cluster_id = find_largest_cluster();

                    // Get data from chosen cluster and erase the cluster from data
                    std::vector<std::pair<int, std::vector<float>>> cluster_data = kmeans_results[largest_cluster_id];
                    cluster_data.erase(cluster_data.begin());

                    // Erase cluster from kmeans_results
                    kmeans_results.erase(kmeans_results.begin() + largest_cluster_id);

                    // Create new clusters
                    std::vector<std::vector<float>> new_clusters = create_clusters();
                    assign_clusters(cluster_data, new_clusters);
                }
            }
                                
        }

        void build_index(const int& max_num_clusters){
            // Run kmeans
            bisect_kmeans(max_num_clusters);

            // Build IVF index
            for(int i = 0; i < static_cast<int>(kmeans_results.size()); ++i){
                std::vector<std::pair<int, std::vector<float>>> data_in_cluster = kmeans_results[i];
                IVF[i] = data_in_cluster;
            }
        }

        std::vector<std::vector<std::pair<int, std::vector<float>>>> get_kmeans_clusters(){
            return kmeans_results;
        }

        std::map<int, std::vector<std::pair<int, std::vector<float>>>> get_IVF(){
            return IVF;
        }

        int get_IVF_size(){
            return IVF.size();

        }


};