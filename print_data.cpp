#include <iostream>
#include "utils/data.hpp"

int main(){
    GraphData<int> gd("data/siftsmall/siftsmall_groundtruth.ivecs");
    std::cout << "Vector dimension: " << gd.get_vector_dim() << std::endl;
    std::cout << "Number of vectors: " << gd.get_num_vectors() << std::endl;
    gd.print_vectors();
}