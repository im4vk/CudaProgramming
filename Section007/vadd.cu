#include <cuda_runtime.h>
#include <iostream>

const int N = 1024 * 1024 * 32; // Size of the vectors

__global__ void vector_add_1_element(float *a, float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    vector_add_1_element<<<gridSize, blockSize>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
