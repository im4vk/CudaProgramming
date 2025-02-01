// 02_wmma.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Matrix size must be multiple of 16 for this simple WMMA example
static const int N = 1024;

//------------------------------------------------------------------------------
// CPU reference multiply (in float), for verifying GPU results
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
// WMMA GEMM kernel: each block computes one 16x16 tile of the output
__global__ void wmmaGemmKernel(const half* A, const half* B, half* C, int N)
{
    // Each block handles one 16×16 tile in the final NxN matrix
    int tileCol = blockIdx.x;
    int tileRow = blockIdx.y;

    // Top-left corner of this tile in the output matrix
    int row = tileRow * 16;
    int col = tileCol * 16;

    // Create an accumulator fragment in FLOAT
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Loop over K dimension in chunks of 16
    for (int k = 0; k < N; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bFrag;

        // Load 16x16 tile from A (row-major)
        wmma::load_matrix_sync(aFrag, A + row*N + k, N);
        // Load 16x16 tile from B (col-major for this example)
        wmma::load_matrix_sync(bFrag, B + k*N + col, N);

        // Multiply-accumulate
        wmma::mma_sync(acc, aFrag, bFrag, acc);
    }

    // Manually convert float accumulators -> half and store
    // wmma::store_matrix_sync() won't do float->half conversion automatically.
    // We'll do it in a simple loop. Since we launched blockDim(1,1), we can do this from one thread.
    // For a 16x16 fragment, acc.num_elements == 256.
    // We'll index the fragment as (i*16 + j).
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float val = acc.x[i * 16 + j];
            C[(row + i)*N + (col + j)] = __float2half(val);
        }
    }
}

//------------------------------------------------------------------------------
int main()
{
    // Host allocations
    size_t size = N * N * sizeof(half);
    half* h_A = (half*)malloc(size);
    half* h_B = (half*)malloc(size);
    half* h_C = (half*)malloc(size);

    // CPU reference in float
    float* h_Ref = (float*)malloc(N * N * sizeof(float));

    // Initialize input matrices
    for(int i = 0; i < N*N; i++){
        float valA = static_cast<float>(rand() % 3);
        float valB = static_cast<float>(rand() % 3);
        h_A[i] = __float2half(valA);
        h_B[i] = __float2half(valB);
    }

    // CPU reference (very slow for 8192; be aware!)
    printf("Running CPU reference multiply (N=%d) ...\n", N);
    cpuReferenceGemmHalf(h_A, h_B, h_Ref, N);

    // Device allocations
    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch WMMA kernel
    // Each block processes one 16x16 tile
    dim3 blockDim(1, 1);          
    dim3 gridDim(N/16, N/16);
    wmmaGemmKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
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

    printf("WMMA GEMM done.\n");
    return 0;
}
