#include <cuda_runtime.h>
#include <iostream>

const int N = 1024 * 1024 * 32;  // Size of the vectors

// Optimized vector addition using vectorized memory access
__global__ void vector_add_vectorized(float4 *a, float4 *b, float4 *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one float4 (which is equivalent to 4 floats at once)
    if (i < N / 4) {
        c[i] = make_float4(a[i].x + b[i].x,
                           a[i].y + b[i].y,
                           a[i].z + b[i].z,
                           a[i].w + b[i].w);
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float4 *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory for vectorized access (float4 type)
    cudaMalloc((void**)&d_a, (N / 4) * sizeof(float4));
    cudaMalloc((void**)&d_b, (N / 4) * sizeof(float4));
    cudaMalloc((void**)&d_c, (N / 4) * sizeof(float4));

    // Copy data to device (cast float pointers to float4 pointers for vectorized access)
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (N / 4 + blockSize - 1) / blockSize;  // Processing float4 instead of float

    // Launch vectorized kernel
    vector_add_vectorized<<<gridSize, blockSize>>>(d_a, d_b, d_c);

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
