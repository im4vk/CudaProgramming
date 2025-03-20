#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define N 1024

__global__ void matMulCUDA(half *A, half *B, half *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        half sum = __float2half(0.0f);
        for (int k = 0; k < N; k++) {
            sum = __hadd(sum, __hmul(A[row * N + k], B[k * N + col]));
        }
        C[row * N + col] = sum;
    }
}

void randomMatrix(half *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
}

int main() {
    size_t bytes = N * N * sizeof(half);

    half *h_A, *h_B, *h_C;
    half *d_A, *d_B, *d_C;

    h_A = (half*)malloc(bytes);
    h_B = (half*)malloc(bytes);
    h_C = (half*)malloc(bytes);

    randomMatrix(h_A, N * N);
    randomMatrix(h_B, N * N);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(N / blockSize.x, N / blockSize.y);

    matMulCUDA<<<gridSize, blockSize>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Matrix multiplication completed using CUDA Cores (FP16).\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

