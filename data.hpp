#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class GraphData {
    private:
        std::string filename;
        int vector_dim;
        std::vector<std::vector<float>> data_vecs;
        std::string a = "abc"; 

    public:
        GraphData(const std::string& file) : 
        filename(file) {
            std::ifstream input(filename, std::ios::binary);

            if (!input) {
                std::cerr << "Error opening file: " << filename << std::endl;
                return;
            }

            while (true) {
                input.read(reinterpret_cast<char*>(&vector_dim), sizeof(int));

                if (input.eof()) {
                    break;
                }

                std::vector<float> elem_vec(vector_dim);
                input.read(reinterpret_cast<char*>(elem_vec.data()), vector_dim * sizeof(int));

                data_vecs.push_back(elem_vec); 
            }
            input.close();
        }

        std::vector<std::vector<float>> get_data(){
            return data_vecs;
        }

        int get_vector_dim() {
            return vector_dim;
        }

        int get_num_vectors() {
            return data_vecs.size();
        }

        void print_vectors(int num_sample = 10) {     
            std::cout << "Number of samples: " << num_sample << std::endl;

            int i = 1;
            for (const auto& vec : data_vecs) {
                std::cout << "Elements: ";
                for (const auto& elem : vec) {
                    std::cout << elem << " ";
                }           
                ++i;
                std::cout << std::endl;
                if(i == num_sample){
                    break;
                }
            }
        }
};

