#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdlib>

#define N 1024

void cpu_matrix_multiply(half* A, half* B, float* C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += __half2float(A[i * N + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    half* h_A = new half[N * N];
    half* h_B = new half[N * N];
    float* h_C = new float[N * N];
    float* h_C_cpu = new float[N * N];
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, N * N * sizeof(half));
    cudaMalloc(&d_B, N * N * sizeof(half));
    cudaMalloc(&d_C, N * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, N * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(half), cudaMemcpyHostToDevice);
    
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, N, N,
                 &alpha,
                 d_A, CUDA_R_16F, N,
                 d_B, CUDA_R_16F, N,
                 &beta,
                 d_C, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cpu_matrix_multiply(h_A, h_B, h_C_cpu);
    
    std::cout << "First 10 elements of GPU result:" << std::endl;
    for (int i = 0; i < 10; i++) std::cout << h_C[i] << " ";
    std::cout << "\n";
    
    std::cout << "First 10 elements of CPU result:" << std::endl;
    for (int i = 0; i < 10; i++) std::cout << h_C_cpu[i] << " ";
    std::cout << "\n";
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}

