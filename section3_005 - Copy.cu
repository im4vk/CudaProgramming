#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024*1024*32  // Define the size of the vectors

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int *A, *B, *C;            // Host vectors
    int *d_A, *d_B, *d_C;      // Device vectors
    int size = SIZE * sizeof(int);

    // CUDA event creation, used for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate and initialize host vectors
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;
    }

    // Allocate device vectors
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy host vectors to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Start recording
    cudaEventRecord(start);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 96;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);

    // Stop recording
    cudaEventRecord(stop);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Calculate and print the execution time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f milliseconds\n", milliseconds);

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
