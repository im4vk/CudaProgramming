#include <iostream>
#include <cuda_runtime.h>

__global__ void reduce_in_place(float* input, int n) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform in-place reduction within each block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();  // Ensure all threads have completed the previous iteration

        if (tid % (2 * stride) == 0 && index + stride < n) {
            // Perform reduction and store the sum back in place
            input[index] += input[index + stride];
        }
    }

    // Store the block's reduced result in the first position of this block's portion
    if (tid == 0) {
        input[blockIdx.x] = input[blockIdx.x * blockDim.x];  // Write the reduced sum for this block
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
    int gridSize = (n + blockSize - 1) / blockSize;  // Number of blocks
    // Calculate the sum using the CPU function for verification
    float total_sum = cpu_reduce(h_input, 1024 * 1024);
    std::cout << "Total sum (CPU): " << total_sum << std::endl;
    // Perform iterative reduction until we have one block left
    reduce_in_place << <gridSize, blockSize >> > (d_input, n);
    cudaDeviceSynchronize();  // Ensure kernel execution completes
    reduce_in_place << <16, blockSize >> > (d_input, 4096);
    cudaDeviceSynchronize();  // Ensure kernel execution completes


    // Final reduction when gridSize is 1
    reduce_in_place << <1, blockSize >> > (d_input, 16);
    cudaDeviceSynchronize(); 
    // Copy the final result back to the host (the sum should be in h_input[0])
    cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Final sum (GPU): " << h_input[0] << std::endl;

    

    // Free memory
    cudaFree(d_input);
    delete[] h_input;

    return 0;
}
