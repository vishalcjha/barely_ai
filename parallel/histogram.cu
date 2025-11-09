#include <cuda_runtime.h>
#include <iostream>
#define BIN_SIZE 6
#define CFACTOR 32
/**
Following code is inspired by my reading from Programming Massively Parallel Processor.
 */
/**
Each thread is responsible for CFACTOR contineous characters. This is good for CPU but bad idea for GPU.
 */
__global__ void historgram_contigeous_kernal(const char* data, unsigned int lenght, int* result) {
    __shared__ unsigned int histo_s[BIN_SIZE];
    for (int i = threadIdx.x; i < BIN_SIZE; i += blockDim.x) {
        histo_s[i] = 0u;
    }
    __suncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int pos = tid * CFACTOR; pos < (tid + 1) * CFACTOR && pos < lenght; pos++) {
        int value = data[pos] - 'a';
        if (value >= 0 && value < 26) {
            // need to synchronize mutating access of shared memory
            atomic_add(&histo_s[value], 1);
        }
    }

    __syncthreads();
    for (unsigned int i = threadIdx.x; i < BIN_SIZE; i+= blockDim.x) {
        if (histo_s[i] > 0) {
            atomic_add(&result[i], histo_s[i]);
        }
    }
}

/**
In this case threads own memory area STRIDE away, which is blockDim.x * gridDim.x.
This is bad memory access pattern for CPU but works perfect for GPU.
 */
__global__ void convergence_kernel(const char* data, unsigned int length, int* result) {
    __shared__ unsigned int histo_s[BIN_SIZE];
    for (int i = threadIdx.x; i < BIN_SIZE; i += blockDim.x) {
        histo_s[i] = 0u;
    }
    __suncthreads();
    int tid = blockIdx.x + blockDim.x + threadIdx.x;
    for (unsigned pos = tid; pos < length; pos += blockDim.x * gridDim.x) {
        int value = data[pos] - 'a';
        if (value >= 0 && value < 26) {
            // need to synchronize mutating access of shared memory
            atomic_add(&histo_s[value], 1);
        }
    }

    __syncthreads();
    for (unsigned int i = threadIdx.x; i < BIN_SIZE; i+= blockDim.x) {
        if (histo_s[i] > 0) {
            atomic_add(&result[i], histo_s[i]);
        }
    }
}