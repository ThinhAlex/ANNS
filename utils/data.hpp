#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

template <typename T>
class GraphData {
    private:
        std::string filename;
        int vector_dim;
        std::vector<std::vector<T>> data_vecs;

    public:
        GraphData(const std::string& file) : 
        filename(file) {
            if(file.find("fvecs") != std::string::npos){
                load_fvecs();
            }else if (file.find("ivecs") != std::string::npos){
                load_ivecs();
            }else{
                std::cerr << "Unsupported file format: " << filename << std::endl;
            }
        }

        void load_fvecs(){
            std::ifstream input(filename, std::ios::binary);

            if (!input) {
                std::cerr << "Error opening file: " << filename << std::endl;
            }

            while (true) {
                input.read(reinterpret_cast<char*>(&vector_dim), sizeof(int));

                if (input.eof()) {
                    break;
                }

                std::vector<T> elem_vec(vector_dim);
                input.read(reinterpret_cast<char*>(elem_vec.data()), vector_dim * sizeof(T));

                data_vecs.push_back(elem_vec); 
            }
            input.close();
        }

        void load_ivecs(){
            std::ifstream input(filename, std::ios::binary);

            if (!input){
                std::cerr << "Error opening file: " << filename << std::endl;
            }

            while(true){
                input.read(reinterpret_cast<char*>(&vector_dim), sizeof(int));

                if (input.eof()){
                    break;
                }

                std::vector<T> elem_vec(vector_dim);
                input.read(reinterpret_cast<char*>(elem_vec.data()), vector_dim * sizeof(T));
                data_vecs.push_back(elem_vec); 

            }
            input.close();

        }

        std::vector<std::vector<T>> get_data(){
            return data_vecs;
        }

        int get_vector_dim() {
            return vector_dim;
        }

        int get_num_vectors() {
            return data_vecs.size();
        }

        void print_vectors(int num_sample = 1) {     
            std::cout << "Number of samples: " << num_sample << std::endl;

            int i = 1;
            for (const auto& vec : data_vecs) {
                std::cout << "Elements: ";
                for (const auto& elem : vec) {
                    std::cout << elem << " ";
                }           
                std::cout << std::endl;
                if(i == num_sample){
                    break;
                }
                ++i;
            }
        }
};

