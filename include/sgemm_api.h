#pragma once

enum class SgemmEnum
{
    Naive,
    Tiled,
    RegisterTiled,
    WMMA,
    CPU
};

void sgemm(const float* A, const float* B, float* C, int M, int N, int K, SgemmEnum type);