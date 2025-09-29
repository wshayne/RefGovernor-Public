#ifndef FILEOPS_REFGOV_H
#define FILEOPS_REFGOV_H

#include <string>
#include <iostream>
#include <fstream>

template<size_t row, size_t col, typename T>
void array_to_csv(T array[row][col], std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    for (size_t i=0; i<row; ++i) {
        for (size_t j=0; j<col; ++j) {
            if (j < (col-1)) {
                outfile << array[i][j] << ",";
            } else {
                outfile << array[i][j] << ";\n";
            }
        }
    }
    outfile.close();
}

template<typename T>
void ptr_array_to_csv(T* arr, std::string filename, size_t row, size_t col) {
    std::ofstream outfile;
    outfile.open(filename);
    for (size_t i=0; i<row; ++i) {
        for (size_t j=0; j<col; ++j) {
            if (j < (col-1)) {
                outfile << arr[i * col + j] << ",";
            } else {
                outfile << arr[i * col + j] << "\n";
            }
        }
    }
    outfile.close();
}

#endif