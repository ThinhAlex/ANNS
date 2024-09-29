#include <iostream>
#include "data.hpp"

int main(){
    GraphData gd("data/siftsmall/siftsmall_base.fvecs");
    std::cout << "Vector dimension: " << gd.get_vector_dim() << std::endl;
    std::cout << "Number of vectors: " << gd.get_num_vectors() << std::endl;
    gd.print_vectors();
}