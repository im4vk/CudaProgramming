#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#define N 2048
#define TILE_SIZE 16 // Using 16x16 matrix tiles

using namespace nvcuda;

__global__ void matMulTensorCores(half *A, half *B, half *C) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    if (warpM < N / TILE_SIZE && warpN < N / TILE_SIZE) {
        wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> fragA;
        wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major> fragB;
        wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, half> fragC;

        wmma::fill_fragment(fragC, __float2half(0.0f));

        for (int k = 0; k < N / TILE_SIZE; k++) {
            wmma::load_matrix_sync(fragA, A + warpM * TILE_SIZE * N + k * TILE_SIZE, N);
            wmma::load_matrix_sync(fragB, B + k * TILE_SIZE * N + warpN * TILE_SIZE, N);
            wmma::mma_sync(fragC, fragA, fragB, fragC);
        }

        wmma::store_matrix_sync(C + warpM * TILE_SIZE * N + warpN * TILE_SIZE, fragC, N, wmma::mem_row_major);
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

    dim3 blockSize(32, 32);
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE);

    matMulTensorCores<<<gridSize, blockSize>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Matrix multiplication completed using Tensor Cores (FP16).\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

