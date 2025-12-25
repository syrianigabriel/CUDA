#include "sgemm.h"
#include "timer.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include "helpers.h"

#define MAX_NUM 10
#define MIN_NUM -10
#define TILE_WIDTH 16

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int N = atoi(argv[1]);

    // initialize matrices A, B, C of size NxN
    std::vector<float> A(N*N), B(N*N), C(N*N);

    // fill matrices with random values
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N + j] = MIN_NUM + static_cast<float>(rand()) / RAND_MAX * (MAX_NUM - MIN_NUM);
            B[i*N + j] = MIN_NUM + static_cast<float>(rand()) / RAND_MAX * (MAX_NUM - MIN_NUM);
        }
    }

    // initalize device pointers (VRAM)
    float* d_A;
    float* d_B;
    float* d_C;

    // allocate memory on device and store addresses in d_X ptrs
    CUDA_CHECK(cudaMalloc((void**) &d_A, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &d_B, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &d_C, N*N*sizeof(float)));

    // copy A and B matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

    CudaTimer timer;

    timer.start();
    // execute on GPU
    sgemm(d_A, d_B, d_C, N, N, N, SgemmKernelEnum::Tiled);
    CUDA_CHECK(cudaGetLastError());
    float tiledSgemmMs = timer.stop();

    timer.start();
    // execute on GPU
    sgemm(d_A, d_B, d_C, N, N, N, SgemmKernelEnum::Naive);
    CUDA_CHECK(cudaGetLastError());
    float naiveSgemmMs = timer.stop();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}