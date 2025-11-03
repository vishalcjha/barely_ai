#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>
#include <chrono>
#include <algorithm>

using DURATION = std::chrono::duration<long long, std::micro>;

/*
A helper method to validate last command was not error.
*/
void cudaCheckError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cout << "CUDA Error:" << msg << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE );
    }
}

/*
Cuda follows C language. This results in cudaMalloc and cudaFree.
Following the RAII pattern, we can wrap this in C++ object.
*/
template <typename T>
struct CudaMem {
    explicit CudaMem(std::size_t len) : len_(len), size_(len * sizeof(T)) {
        cudaCheckError(cudaMalloc(&mem_, size_), "Failed to allocate memory");
    }

    void copyToDevice(const T* host) {
        cudaCheckError(cudaMemcpy(mem_, host, size_, cudaMemcpyHostToDevice),
        "Failed to copy from host to device");
    }

    void copyToHost(T* host) {
        cudaCheckError(cudaMemcpy(host, mem_, size_, cudaMemcpyDeviceToHost),
            "Failed to copy from device to host");
    }

    T* getMem() {
        return mem_;
    }
    private:
    T* mem_;
    std::size_t len_;
    std::size_t size_;
};

__global__ void matrixMul(float* first, float* second, float* result, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float res = 0;
        for (int i = 0; i < K; i++) {
            int fIndex = row * K + i;
            int sIndex = i * N + col;
            res += first[fIndex] * second[sIndex];
        }
        result[row * N + col] = res;
    }
}

void matrixMulCPU(const float* first, const float* second, float* third, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                third[i * N + j] += first[i * K + k] + second[k * N + j];
            }
        }
    }
}

void fillWithRandomNumber(std::vector<float>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::generate(vec.begin(), vec.end(), [&]() {return dis(gen);});
}

struct Runner {
    static DURATION time(int warmRunCount, int testRunCount, std::function<void()> run) {
        for (int i = 0; i < warmRunCount; i++) {
          run();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < testRunCount; i++) {
            run();
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
};

int main() {
    constexpr dim3 BLOCK_DIM = {256, 256};
    int M = 1000, N = 1500, K = 1200; 
    std::vector<float>host_a(M * K), host_b(K * N), host_c(M * N);
    fillWithRandomNumber(host_a);
    fillWithRandomNumber(host_b);
    fillWithRandomNumber(host_c);

    CudaMem<float> dev_a(M * K), dev_b(K * N), dev_c(M * N);
    dev_a.copyToDevice(&host_a[0]);
    dev_b.copyToDevice(&host_b[0]);

    // basically a ceil method to make sure each element of result matrix aks C is covered.
    dim3 GRID_DIM = {(M + BLOCK_DIM.x -1) / BLOCK_DIM.x, (N  + BLOCK_DIM.y -1) / BLOCK_DIM.y};

    auto cpu_duration = Runner::time(1, 10, [&] () { matrixMulCPU(&host_a[0], &host_b[0], &host_c[0], M, N, K);});
    std::cout << "10 CPU matrix multiplication took " << cpu_duration.count() << " microseconds" << std::endl;

    auto gpu_duration_with_memcpy = Runner::time(1,
        10,
        [&] () { 
            dev_a.copyToDevice(&host_a[0]);
            dev_b.copyToDevice(&host_b[0]);
            matrixMul<<<GRID_DIM, BLOCK_DIM>>>(dev_a.getMem(), dev_b.getMem(), dev_c.getMem(), M, N, K);
            cudaDeviceSynchronize();
            dev_c.copyToHost(&host_c[0]);
        }
    );
    std::cout << "10 GPU matrix multiplication with copy took " << gpu_duration_with_memcpy.count() << " microseconds" << std::endl;

    auto gpu_duration = Runner::time(1, 10, [&] () { 
        matrixMul<<<GRID_DIM, BLOCK_DIM>>>(dev_a.getMem(), dev_b.getMem(), dev_c.getMem(), M, N, K);
        cudaDeviceSynchronize();
    });
    std::cout << "10 GPU matrix multiplication took " << gpu_duration.count() << " microseconds" << std::endl;

    return EXIT_SUCCESS;
}