#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_NUM 10
#define MIN_NUM -10
#define TILE_WIDTH 16

using namespace std;

void cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%d) at %s:%d\n",
                cudaGetErrorString(err), err, file, line);
        exit(err);
    }
}

#define CUDA_CHECK(call) cudaCheck(call, __FILE__, __LINE__)

__global__ void sq_mat_mul_kernel_tiled(float* A, float* B, float* C, int N)
{
    // Thread information
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Corresponding to entry C[i, j]
    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;

    // Declare block-level shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Result of C[i,j]
    float sum = 0;

    // Split dot product into tiles
    for (int phase = 0; phase < ceil((float) N/TILE_WIDTH); phase++)
    {
        // A tiles move from left to right
        if (i < N && (phase*TILE_WIDTH+tx) < N)
            sh_A[ty][tx] = A[i*N + (phase*TILE_WIDTH+tx)];
        else
            sh_A[ty][tx] = (float) 0;

        // B tiles move from top to bottom
        if (j < N && (phase*TILE_WIDTH+ty) < N)
            sh_B[ty][tx] = B[(phase*TILE_WIDTH+ty)*N + j];
        else
            sh_B[ty][tx] = 0;
        
        __syncthreads();

        // Compute partial dot product of matrices in shared memory
        for (int k = 0; k < TILE_WIDTH; k++)
            sum += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }

    // Fill in value in result matrix
    if (i < N && j < N)
        C[i*N + j] = sum;
}

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <matrix_size>\n";
        return 1;
    }

    int N = atoi(argv[1]);

    // initialize matrices A, B, C of size NxN
    float* A = (float*) malloc(N*N*sizeof(float));
    float* B = (float*) malloc(N*N*sizeof(float));
    float* C = (float*) malloc(N*N*sizeof(float));

    // fill matrices with random values
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N + j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            B[i*N + j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
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
    CUDA_CHECK(cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice));

    // block size (x, y, z)
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((N + TILE_WIDTH - 1)/TILE_WIDTH, (N + TILE_WIDTH - 1)/TILE_WIDTH);

    float time;
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, 0));
    // execute on GPU
    sq_mat_mul_kernel_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(end, 0));

    // wait for all kernels to finish
    CUDA_CHECK(cudaEventSynchronize(end));

    cudaEventElapsedTime(&time, start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    // copy result back to host memory
    CUDA_CHECK(cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    cout << N << " " << time << "\n";

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}