#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda::wmma;

// Define matrix dimension N (must be multiple of 16)
#ifndef N
#define N 1024
#endif

// We will use 16×16 WMMA tiles
#define TILE_SIZE 16

// Kernel: half-precision inputs, float accumulation/output
//  1D block of 128 threads => 4 warps per block
//  Each warp computes one 16×16 tile in the output
__global__ void matMulTensorCoresFp32(const half* A, const half* B, float* C, int n)
{
    // warpsPerBlock = 4 if blockDim.x=128
    int warpsPerBlock = blockDim.x / 32;
    // global warp ID
    int warpId = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);

    int tilesPerDim = n / TILE_SIZE;               // e.g. N=256 => 16
    int tileCount   = tilesPerDim * tilesPerDim;   // e.g. 16*16=256

    if (warpId >= tileCount) return;

    // Identify which 16×16 tile we're responsible for
    int tileRow = warpId / tilesPerDim;
    int tileCol = warpId % tilesPerDim;

    // Create WMMA fragments
    // matrix_a uses half in col_major
    fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, col_major> aFrag;
    // matrix_b uses half in row_major
    fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, row_major> bFrag;
    // accumulator uses float
    fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> cFrag;

    // Initialize accumulator to 0.f
    fill_fragment(cFrag, 0.0f);

    // Loop over the K dimension in steps of 16
    for (int k0 = 0; k0 < n; k0 += TILE_SIZE)
    {
        const half* Ablock = A + (tileRow * TILE_SIZE * n) + k0;
        const half* Bblock = B + (k0 * n) + (tileCol * TILE_SIZE);

        load_matrix_sync(aFrag, Ablock, n);
        load_matrix_sync(bFrag, Bblock, n);

        // Multiply-accumulate
        mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    // Write the 16×16 tile of results to the float matrix C
    float* Csub = C + (tileRow * TILE_SIZE * n) + (tileCol * TILE_SIZE);
    store_matrix_sync(Csub, cFrag, n, mem_row_major);
}


// Simple helper: fill a half-precision matrix with random values [0..1]
void randomMatrix(half *mat, int size) {
    for (int i = 0; i < size; i++) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        mat[i] = __float2half(r);
    }
}

// CPU reference multiply, but store result in float
//   C[i,j] = sum_{k} ( float(A[i,k]) * float(B[k,j]) )
void cpuMultiplyHalfFloat(const half* A, const half* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                float aVal = __half2float(A[i * n + k]);
                float bVal = __half2float(B[k * n + j]);
                sum += aVal * bVal;
            }
            C[i * n + j] = sum;
        }
    }
}



int main()
{
    // N must be multiple of 16
    if (N % TILE_SIZE != 0) {
        std::cerr << "Error: N must be a multiple of 16.\n";
        return 1;
    }

    srand(0);

    // Host memory: half inputs, float output
    size_t totalElements = N * N;
    size_t bytesAorB     = totalElements * sizeof(half);
    size_t bytesC        = totalElements * sizeof(float);

    half*  h_A = (half*) malloc(bytesAorB);
    half*  h_B = (half*) malloc(bytesAorB);
    float* h_C = (float*) malloc(bytesC);    // GPU result
    float* h_Ccpu = (float*) malloc(bytesC); // CPU reference

    // Random initialization of A, B in half
    randomMatrix(h_A, N * N);
    randomMatrix(h_B, N * N);

    // Allocate device memory
    half*  d_A = nullptr;
    half*  d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, bytesAorB);
    cudaMalloc(&d_B, bytesAorB);
    cudaMalloc(&d_C, bytesC);

    // Copy A, B to device
    cudaMemcpy(d_A, h_A, bytesAorB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesAorB, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytesC);

    // Compute grid/block configuration
    //  - 128 threads => 4 warps per block
    //  - total 16×16 tiles = (N/16)*(N/16)
    //  - each block handles 4 tiles
    int tilesPerDim  = N / TILE_SIZE;
    int tileCount    = tilesPerDim * tilesPerDim;
    int warpsPerBlk  = 128 / 32;  // 4
    int blocksNeeded = (tileCount + warpsPerBlk - 1) / warpsPerBlk;

    dim3 blockSize(128);
    dim3 gridSize(blocksNeeded);

    // Launch kernel
    matMulTensorCoresFp32<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    // CPU reference multiply (in float)
    cpuMultiplyHalfFloat(h_A, h_B, h_Ccpu, N);

    // Compare first 10 elements
    std::cout << "Matrix multiplication (FP16 inputs -> FP32 output) completed.\n";
    std::cout << "Compare first 10 elements (CPU vs. GPU):\n";
    for (int i = 0; i < 10; i++) {
        float cpuVal = h_Ccpu[i];
        float gpuVal = h_C[i];
        std::cout << "C[" << i << "]: CPU=" << cpuVal
                  << ", GPU=" << gpuVal << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Ccpu);

    return 0;
}
