#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device;
    cudaGetDevice(&device); // Get current CUDA device
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, device);
    printf("Max_threads_per_SM  0: %d \n", prop.maxThreadsPerMultiProcessor);
    printf("Max_warps_per_SM    0: %d \n\n\n", (prop.maxThreadsPerMultiProcessor)/32);




    int maxThreadsPerMP = 0;
    cudaDeviceGetAttribute(&maxThreadsPerMP, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    printf("Max_threads_per_SM  1: %d   \n", maxThreadsPerMP);
    printf("Max_warps_per_SM    1: %d   \n", maxThreadsPerMP/32);

    return 0;
}
