#!/usr/bin/env python3
import sys, requests, urllib.parse

# If a filename is provided, read it; otherwise use the fixed code from your payload
if len(sys.argv) == 2:
    src = open(sys.argv[1], 'r', encoding='utf-8').read()

else:
    THREADS_PER_BLOCK = 4
    src = r'''
    #include <cuda_runtime.h>
    #include "device_launch_parameters.h"
    #include <stdio.h>

    __global__ void test01(){
        printf("block %d  thread %d\n", blockIdx.x, threadIdx.x);
    }

    int main(){
        int n=0; cudaGetDeviceCount(&n);
        if (!n) { puts("[no CUDA GPU detected]"); return 0; }

        test01<<<2,4>>>();                    // no spaces
        cudaError_t e1 = cudaGetLastError();
        cudaError_t e2 = cudaDeviceSynchronize();
        printf("launch = %s  sync = %s\n",
            cudaGetErrorString(e1), cudaGetErrorString(e2));
        return 0;
    }
    '''

compiler = "nvcc125u1" 
#compiler = "nvcc129"
# print(src)

data = {
    "lang": "cuda",
    "filterAnsi": "true",
    "execute": "true",
    "labels": "true",
    "libraryCode": "true",
    "directives": "true",
    "commentOnly": "true",
    "compiler": compiler,
    "userArguments": "",
    "source": src,
}

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://godbolt.org",
    "Referer": "https://godbolt.org/noscript/cuda",
    "Accept": "text/plain; charset=utf-8",
    "User-Agent": "curl/8.x"
}

r = requests.post("https://godbolt.org/api/noscript/compile",data=data, headers=headers)

print(r.text)
