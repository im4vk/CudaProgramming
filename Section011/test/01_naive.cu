// naive_gemm.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Matrix size
static const int N = 1024;

//------------------------------------------------------------------------------
// CPU reference multiply: C_ref = A * B in float
// A and B are in half precision, C_ref is in float.
void cpuReferenceGemmHalf(const half* A, const half* B, float* C_ref, int N) {
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < N; k++){
                float a_ij = __half2float(A[i*N + k]);
                float b_jk = __half2float(B[k*N + j]);
                sum += a_ij * b_jk;
            }
            C_ref[i*N + j] = sum;
        }
    }
}

//------------------------------------------------------------------------------
// Compare GPU result (in half) to CPU reference (in float)
// We do an element-wise check with a tolerance.
bool compareResultsHalfToFloat(const half* GPU_result, const float* CPU_result, 
                               int N, float tolerance = 1e-3f) {
    for(int i = 0; i < N*N; i++){
        float gpu_val = __half2float(GPU_result[i]);
        float cpu_val = CPU_result[i];
        float diff = fabs(gpu_val - cpu_val);
        float relative = diff / (fabs(cpu_val) + 1e-7f);

        if (diff > tolerance && relative > tolerance) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n", 
                   i, gpu_val, cpu_val, diff);
            return false;
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// Naive GPU kernel: each thread computes one element of C by iterating over K
__global__ void naiveGemmKernel(const half* A, const half* B, half* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            // Convert half->float, multiply, accumulate
            sum += __half2float(A[row * N + k]) * __half2float(B[k * N + col]);
        }
        // Convert float->half
        C[row * N + col] = __float2half(sum);
    }
}

//------------------------------------------------------------------------------
int main()
{
    // Allocate host memory
    size_t size = N * N * sizeof(half);
    half* h_A = (half*)malloc(size);
    half* h_B = (half*)malloc(size);
    half* h_C = (half*)malloc(size);

    // For CPU reference, store result in float
    float* h_Ref = (float*)malloc(N * N * sizeof(float));

    // Initialize A and B
    for (int i = 0; i < N*N; i++) {
        float valA = static_cast<float>(rand() % 3);
        float valB = static_cast<float>(rand() % 3);
        h_A[i] = __float2half(valA);
        h_B[i] = __float2half(valB);
    }

    // CPU reference (WARNING: extremely slow for N=8192 in single-thread!)
    printf("Running CPU reference multiply ...\n");
    cpuReferenceGemmHalf(h_A, h_B, h_Ref, N);

    // Allocate device memory
    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch naive kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1)/block.x, (N + block.y - 1)/block.y);
    naiveGemmKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Compare to CPU reference
    printf("Comparing GPU result to CPU reference...\n");
    bool pass = compareResultsHalfToFloat(h_C, h_Ref, N);
    if (!pass) {
        printf("ERROR: Results do not match!\n");
    } else {
        printf("PASS: GPU results match CPU reference.\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Ref);

    printf("Naive GEMM done.\n");
    return 0;
}

