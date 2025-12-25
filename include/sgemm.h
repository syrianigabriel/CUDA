#pragma once

enum class SgemmKernelEnum
{
    Naive,
    Tiled,
    RegisterTiled,
    WMMA
};

void sgemm(const float* A, const float* B, float* C, int M, int N, int K, SgemmKernelEnum kernel);