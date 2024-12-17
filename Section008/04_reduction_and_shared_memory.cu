#include <iostream>
#include <cuda_runtime.h>

__global__ void reduce_in_place_shared_memory(float* input, int n) {
    __shared__ float shared[256];  // Shared memory array for this block

    int tid = threadIdx.x;
    int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // Load two elements from global memory into shared memory, if within bounds  - shared[tid] = input[index] + input[index + blockDim.x];
    shared[tid] = (index < n ? input[index] : 0.0f) + (index + blockDim.x < n ? input[index + blockDim.x] : 0.0f);
    __syncthreads();  // Ensure all threads have loaded their data into shared memory

    // Perform in-place reduction within shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();  // Ensure all threads have completed the current iteration
    }

    // Store the block's reduced result in the first position of this block's portion in global memory
    if (tid == 0) {
        input[blockIdx.x] = shared[0];  // Write the reduced sum for this block to global memory
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
    size_t sharedMemSize = blockSize * sizeof(float);  // Shared memory size per block

    // Calculate the sum using the CPU function for verification
    float total_sum = cpu_reduce(h_input, n);
    std::cout << "Total sum (CPU): " << total_sum << std::endl;

    // Perform iterative reduction until we have one block left
    while (gridSize > 1) {
        reduce_in_place_shared_memory << <gridSize, blockSize, sharedMemSize >> > (d_input, n);
        cudaDeviceSynchronize();  // Ensure kernel execution completes

        // Update n to reflect the reduced number of elements
        n = gridSize;
        gridSize = (n + 2 * blockSize - 1) / (2 * blockSize);  // Update gridSize for the next iteration
    }

    // Final reduction when gridSize == 1
    reduce_in_place_shared_memory << <1, blockSize, sharedMemSize >> > (d_input, n);
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
