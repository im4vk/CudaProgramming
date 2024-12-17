#include <iostream>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduce_with_warp_optimization(float* input, int n) {
    extern __shared__ float shared[];  // Shared memory for inter-warp reduction
    int tid = threadIdx.x;
    int index = 2 * blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    // Load two elements from global memory into registers (to minimize shared memory usage)
    // This is just to use double size for each vector (512 elements for a block of 256 threads) (more work per thread optimization)
    //once you finish this, you will have 256 elements to work on in the next step.
    sum = (index < n ? input[index] : 0.0f) + (index + blockDim.x < n ? input[index + blockDim.x] : 0.0f);
    // Perform warp-level reduction
    for (int offset = warpSize >>1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    //Now, you have finished the first step, that means that you calculated one partial sum per warp.
    // we need to store these values in the shared memory
    //and then we need to add all of these partial sums together to get a final sum (one element per block From all warps)
    //The final output here is all partial sums will be in the first warp
    if (tid % warpSize == 0) {
        shared[tid / warpSize] = sum;
    }
    __syncthreads();  // Synchronize threads before inter-warp reduction
    //add all partial sums together (from the first warp)
    if (tid < warpSize) {
        sum = (tid < (blockDim.x / warpSize)) ? shared[tid] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    }
    // Store the final reduced result for the block in global memory
    if (tid == 0) {
        input[blockIdx.x] = sum;
    }
}

float cpu_reduce(float* input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    return sum;
}

int main() {
    int n = 1024 * 1024;  // Number of elements
    size_t bytes = n * sizeof(float);

    // Host memory allocation
    float* h_input = new float[n];
    float* d_input;

    // Initialize input array
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i + 1);  // Initialize from 1 to n
    }

    // Device memory allocation
    cudaMalloc(&d_input, bytes);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch the reduction kernel multiple times
    int blockSize = 256;  // Number of threads per block
    int gridSize = (n + 2 * blockSize - 1) / (2 * blockSize);  // Number of blocks
    size_t sharedMemSize = (blockSize / 32) * sizeof(float);  // Using 32 instead of warpSize

    // Calculate the sum using the CPU function for verification
    float total_sum = cpu_reduce(h_input, n);
    std::cout << "Total sum (CPU): " << total_sum << std::endl;

    // Perform iterative reduction until we have one block left
    while (gridSize > 1) {
        reduce_with_warp_optimization<<<gridSize, blockSize, sharedMemSize>>>(d_input, n);
        cudaDeviceSynchronize();  // Ensure kernel execution completes

        // Update n to reflect the reduced number of elements
        n = gridSize;
        gridSize = (n + 2 * blockSize - 1) / (2 * blockSize);  // Update gridSize for the next iteration
    }

    // Final reduction when gridSize == 1
    reduce_with_warp_optimization<<<1, blockSize, sharedMemSize>>>(d_input, n);
    cudaDeviceSynchronize();  // Ensure final kernel execution completes

    // Copy the final result back to the host (the sum should be in h_input[0])
    cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the final result
    std::cout << "Final sum (GPU): " << h_input[0] << std::endl;

    // Free memory
    cudaFree(d_input);
    delete[] h_input;

    return 0;
}
