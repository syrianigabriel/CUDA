#include "sgemm.h"
#include "kernel_config.h"
#include <stdexcept>

void sgemm(const float* A, const float* B, float* C, int M, int N, int K, SgemmKernelEnum kernel)
{
    if (!A || !B || !C)
    {
        throw std::runtime_error("Null pointer passed to SGEMM!");
    }

    switch (kernel)
    {
        case SgemmKernelEnum::Naive:
            launch_naive_sgemm(A, B, C, M, N, K);
            break;

        case SgemmKernelEnum::Tiled:
            launch_tiled_sgemm(A, B, C, M, N, K);
            break;
        default:
            throw std::runtime_error("Unsupported SGEMM kernel!");
    }
}