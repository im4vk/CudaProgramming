#include <iostream>
#include <cuda_runtime.h>

__global__ void reduce_in_place(float* input, int n) {
    __shared__ float shared[1024];  // Shared memory array for this block
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements from global memory to shared memory
        shared[tid] = input[index];

    __syncthreads();  // Synchronize all threads in the block

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();  // Ensure all threads have completed the previous iteration
        //int x = 2 * stride * tid;
        if (index+ stride < n) {
            shared[tid] += shared[tid + stride];
        }
    }
    
    // Write the block's result to global memory
    if (tid == 0) {
        input[blockIdx.x] = shared[0];
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
    int n = 1024*1024;  // Number of elements
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
    std::cout << "grid size is   " << gridSize << std::endl;
    float total_sum = cpu_reduce(h_input, n);
    std::cout << "Total sum (CPU): " << total_sum << std::endl;
    // Perform iterative reduction until we have one block left
    while (gridSize > 1) {
        reduce_in_place << <gridSize, blockSize >> > (d_input, n);
        cudaDeviceSynchronize();  // Ensure kernel execution completes

        // Update n to reflect the reduced number of elements
        n = gridSize;
        gridSize = (n + blockSize - 1) / blockSize;  // Update gridSize for the next iteration
    }

    // Final reduction when gridSize == 1
    reduce_in_place << <1, blockSize >> > (d_input, n);
    cudaDeviceSynchronize();  // Ensure final kernel execution completes

    // Copy the final result back to the host (the sum should be in h_input[0])
    cudaMemcpy(h_input, d_input, 4*sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Final sum (GPU): " << h_input[0] << std::endl;




    // Free memory
    cudaFree(d_input);
    delete[] h_input;

    return 0;
}
