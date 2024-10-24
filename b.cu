#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024*1024*1024*4  // Define the size of the vectors

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, long long  n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int *A, *B, *C;            // Host vectors
    int *d_A, *d_B, *d_C;      // Device vectors
    //long long size = SIZE * sizeof(int);
    const long long size1 = 1024LL * 1024 * 1024 *3;
const long long size = size1 * sizeof(long);
    // CUDA event creation, used for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;

    // Allocate and initialize host vectors
    
err=cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed0: %s\n", cudaGetErrorString(err));
    }
    err=cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed1: %s\n", cudaGetErrorString(err));
    }
    // Allocate device vectors
    err=cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed2: %s\n", cudaGetErrorString(err));
    }
A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    for (int i = 0; i < size1; i++) {
        A[i] = i;
        B[i] = i+2;
    }

    // Copy host vectors to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Start recording
    cudaEventRecord(start);
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 96;
    int blocksPerGrid = (size1 + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size1);

    // Stop recording
    cudaEventRecord(stop);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Calculate and print the execution time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f milliseconds\n", milliseconds);
    for(int i=0;i<10;i++){
        printf("A=%d\tB=%d -------> C=%d  \n",A[i],B[i],C[i]);
    }
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
