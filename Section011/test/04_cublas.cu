// cublas_gemm.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Matrix size
static const int N = 1024;

//------------------------------------------------------------------------------
// CPU reference multiply
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
int main()
{
    size_t size = N * N * sizeof(half);
    half* h_A = (half*)malloc(size);
    half* h_B = (half*)malloc(size);
    half* h_C = (half*)malloc(size);

    // CPU ref in float
    float* h_Ref = (float*)malloc(N * N * sizeof(float));

    // Init host data
    for (int i = 0; i < N*N; i++){
        float valA = static_cast<float>(rand() % 3);
        float valB = static_cast<float>(rand() % 3);
        h_A[i] = __float2half(valA);
        h_B[i] = __float2half(valB);
    }

    // CPU reference
    printf("Running CPU reference multiply ...\n");
    cpuReferenceGemmHalf(h_A, h_B, h_Ref, N);

    // Device memory
    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Enable Tensor Core math
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha = 1.0f;
    float beta  = 0.0f;

    // N x N = (N x N) * (N x N)
    // C = A * B in half precision, with float alpha/beta
    cublasStatus_t stat = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_A, CUDA_R_16F, N,
        d_B, CUDA_R_16F, N,
        &beta,
        d_C, CUDA_R_16F, N,
        CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasGemmEx failed\n");
    }

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Check results
    printf("Comparing GPU result to CPU reference...\n");
    bool pass = compareResultsHalfToFloat(h_C, h_Ref, N);
    if (!pass) {
        printf("ERROR: Results do not match!\n");
    } else {
        printf("PASS: GPU results match CPU reference.\n");
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Ref);

    printf("cuBLAS GEMM done.\n");
    return 0;
}

