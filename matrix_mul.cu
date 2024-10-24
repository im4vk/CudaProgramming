#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_WIDTH 32

// CUDA kernel to perform matrix multiplication
__global__ void matrixMul(int *a, int *b, int *c, int width) {
    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int Cvalue = 0;

    for (int t = 0; t < width / TILE_WIDTH; ++t) {
        ds_A[ty][tx] = a[Row * width + t * TILE_WIDTH + tx];
        ds_B[ty][tx] = b[(t * TILE_WIDTH + ty) * width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        __syncthreads();
    }

    c[Row * width + Col] = Cvalue;
}

// CPU function to allocate memory and initialize matrices
void matrixMulCPU(int *a, int *b, int *c, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            a[i * width + j] = i * width + j;
            b[i * width + j] = j * width + i;
            c[i * width + j] = 0;
        }
    }
}

// Function to get current time in microseconds
long long getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec;
}

int main() {
    int *a, *b, *c;  // Matrices on the host
    int *d_a, *d_b, *d_c;  // Matrices on the device
    int width = 4096*4;
    int size = width * width * sizeof(int);

    // Allocate memory on host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Allocate memory on device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Initialize matrices
    matrixMulCPU(a, b, c, width);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    // Start timing
    long long start_time = getCurrentTime();

    // Launch kernel
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Stop timing
    long long end_time = getCurrentTime();
    double execution_time = (double)(end_time - start_time) / 1000000.0; // Convert to seconds

    printf("Matrix multiplication execution time: %.6f seconds\n", execution_time);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}
