#pragma once

void launch_naive_sgemm(const float* A, const float* B, float* C, int M, int N, int K);

void launch_double_buffered_sgemm(const float* A, const float* B, float* C, int M, int N, int K);

void launch_tiled_sgemm(const float* A, const float* B, float* C, int M, int N, int K);