#include <stdio.h>
#include <cstdlib>  // for rand()
#define N 4096  // Number of rows
#define M 4096  // Number of columns

// CUDA kernel for element-wise matrix addition
__global__ void matrixAdd(float* A, float* B, float* C, int NN, int MM) {
    // Calculate the row and column indices of the matrix element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (row < NN && col < MM) {
        // Calculate the 1D index for the 2D element
        int index = row * M + col;
        // Perform element-wise addition
        C[index] = A[index] + B[index];
    }
}

int main() {
    // Matrix dimensions
    int size = N * M * sizeof(float);
    
    // Allocate host memory for matrices A, B, and C
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize matrices A and B
// Randomly initialize matrices A and B
    for (int i = 0; i < N * M; i++) {
        h_A[i] = 10*static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  // Random values between 0 and 1
        h_B[i] = 20*static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  // Random values between 0 and 1
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // Copy matrices A and B to the device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(32, 32);  // 16x16 threads per block
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    cudaFuncSetCacheConfig(matrixAdd, cudaFuncCachePreferL1);

    // Launch the kernel
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M);

    // Copy result matrix C back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Check a few results
    printf("A[i] + B[i] = c[i]  \n");
    printf("%f   + %f   = %f   \n", h_A[0],h_B[0],h_C[0]);
    printf("%f   + %f   = %f   \n", h_A[1],h_B[1],h_C[1]);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
