#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <cuda_runtime.h>

void cudaCheckError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cout << "CUDA Error:" << msg << " _ " << cudaGet
    }
}

#endif