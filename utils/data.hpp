#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <new>    

template <typename T>
class GraphData {
private:
    std::string filename;
    int vector_dim = 0;   
    int num_vectors = 0;  
    T* data_vecs = nullptr;

public:
    GraphData(const std::string& file) : filename(file) {
        if (file.find("fvecs") != std::string::npos) {
            load_fvecs();
        } else if (file.find("ivecs") != std::string::npos) {
            load_ivecs();
        } else {
            std::cerr << "Unsupported file format: " << filename << std::endl;
        }
    }

    ~GraphData() {
        free(data_vecs); 
    }

    void load_fvecs() {
        std::ifstream input(filename, std::ios::binary);
        if (!input) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        while (input.peek() != EOF) {
            int current_dim;
            input.read(reinterpret_cast<char*>(&current_dim), sizeof(int));
            input.ignore(current_dim * sizeof(T)); 
            num_vectors++; 
            vector_dim = current_dim; 
        }

        data_vecs = static_cast<T*>(aligned_alloc(32, num_vectors * vector_dim * sizeof(T)));
        if (!data_vecs) throw std::bad_alloc(); 

        input.clear();
        input.seekg(0, std::ios::beg);

        for (int i = 0; i < num_vectors; ++i) {
            input.read(reinterpret_cast<char*>(&vector_dim), sizeof(int));
            input.read(reinterpret_cast<char*>(&data_vecs[i * vector_dim]), vector_dim * sizeof(T));
        }

        input.close();
    }

    void load_ivecs() {
        std::ifstream input(filename, std::ios::binary);
        if (!input) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        while (input.peek() != EOF) {
            int current_dim;
            input.read(reinterpret_cast<char*>(&current_dim), sizeof(int));
            input.ignore(current_dim * sizeof(T));
            num_vectors++; 
            vector_dim = current_dim; 
        }

        data_vecs = static_cast<T*>(aligned_alloc(32, num_vectors * vector_dim * sizeof(T)));
        if (!data_vecs) throw std::bad_alloc();

        input.clear();
        input.seekg(0, std::ios::beg);

        for (int i = 0; i < num_vectors; ++i) {
            input.read(reinterpret_cast<char*>(&vector_dim), sizeof(int));
            input.read(reinterpret_cast<char*>(&data_vecs[i * vector_dim]), vector_dim * sizeof(T));
        }

        input.close();
    }

    T* get_data() {
        return data_vecs;
    }

    int get_vector_dim() const {
        return vector_dim;
    }

    int get_num_vectors() const {
        return num_vectors;
    }

    void print_vectors(int num_sample = 10) const {     
        std::cout << "Number of samples: " << num_sample << std::endl;

        for (int i = 0; i < std::min(num_sample, num_vectors); ++i) {
            std::cout << "Elements " << i << ": ";
            for (int j = 0; j < vector_dim; ++j) {
                std::cout << data_vecs[i * vector_dim + j] << " "; 
            }           
            std::cout << std::endl << std::endl;
        }
    }
};
