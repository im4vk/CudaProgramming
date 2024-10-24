#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuKernelCheck() { gpuKernelAssert(__FILE__, __LINE__); }
inline void gpuKernelAssert(const char *file, int line, bool abort=true) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[ix] = A[ix] + B[ix];
    }
}

int main(int argc, char **argv) {
    // Check if block size is provided as a command line argument
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <block size>\n", argv[0]);
        return 1;
    }

    // Parse block size from command line
    int threadsPerBlock = atoi(argv[1]);
    if (threadsPerBlock <= 0) {
        fprintf(stderr, "Block size must be a positive integer.\n");
        return 1;
    }

    int *A, *B, *C;            // Host vectors
    int *d_A, *d_B, *d_C;      // Device vectors
    long long SIZE = 1024LL * 1024 * 32;
    long size = SIZE * sizeof(int);

    // CUDA event creation, used for timing
    // Allocate device vectors
    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));

    // Allocate and initialize host vectors
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;
    }

    // Copy host vectors to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);
    gpuKernelCheck();

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Calculate and print the execution time
    cudaEventSynchronize(stop);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

