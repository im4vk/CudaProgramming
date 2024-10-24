
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void test01()
{
    //print the blocks and threads IDs
    // warp=32threads. (128 threads/block) --> ( 128/32 = 4 warps/block)
    int warp_ID_Value = 0;
    warp_ID_Value = threadIdx.x / 32;//%
    printf("The block ID is %d --- The thread ID is %d --- The warp ID %d\n",blockIdx.x,threadIdx.x, warp_ID_Value);
}

int main()
{   //add two vectors has 2048 elements
    // kernel_name<<<num_of_blocks , Num_of_threads_per_block>>>();
    //test01 <<<1, 2048>>> ();---- warps/block=2warps, total warps/GPU=4warps
    
    test01 <<<2, 64 >>> ();
    cudaDeviceSynchronize();
    return 0;
}
