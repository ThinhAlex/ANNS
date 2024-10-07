#pragma once
#include <vector>


class Recall{
    private:
        std::vector<std::vector<int>> gt; // groundtruth
        std::vector<std::vector<std::pair<int, double>>> knn_results;

        int num_query = gt.size();
        int gt_k = gt[0].size();
        int query_k = knn_results[0].size();
        double recall = 0;

    public:
        Recall(const std::vector<std::vector<int>>& gt, const std::vector<std::vector<std::pair<int, double>>>& knn_results) :
        gt{gt}, knn_results{knn_results} {
            if (query_k <= gt_k){
                for (int i = 0; i < num_query; ++i){
                    int correct_count = 0;
                    for (int j = 0; j < query_k; ++j){
                        int predict_id = knn_results[i][j].first;
                        int gt_id = gt[i][j];
                        if (predict_id == gt_id){
                            ++correct_count;
                        }
                    }
                    recall += static_cast<double>(correct_count)/query_k;
                }
                recall = recall/num_query;
            }else{
                for (int i = 0; i < num_query; ++i){
                    int correct_count = 0;
                    for (int j = 0; j < gt_k; ++j){
                        int predict_id = knn_results[i][j].first;
                        int gt_id = gt[i][j];
                        if (predict_id == gt_id){
                            ++correct_count;
                        }
                    }
                    recall += static_cast<double>(correct_count)/gt_k;
                }
                recall = recall/num_query;
            }
        }

        double get_recall(){
            return recall;
        }
        
};