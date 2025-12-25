#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include "helpers.h"

class CudaTimer
{
private:
    cudaEvent_t startEvent, endEvent;
public:
    CudaTimer()
    {
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&endEvent));
    }

    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(endEvent));
    }

    void start()
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(startEvent, 0));
    }

    float stop()
    {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(endEvent, 0));
        CUDA_CHECK(cudaEventSynchronize(endEvent));
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, endEvent));
        return ms;
    }
}; 