#!/usr/bin/env python3
import sys, requests, urllib.parse

# If a filename is provided, read it; otherwise use the fixed code from your payload
if len(sys.argv) == 2:
    src = open(sys.argv[1], 'r', encoding='utf-8').read()
else:
    src = '''
    #include <cuda_runtime.h>
    #include "device_launch_parameters.h"
    #include <cstdio>

    __global__ void test01(){
        printf("block id %d --- thread id %d\n", blockIdx.x, threadIdx.x);
    }

    int main(){
        test01<<<1,4>>>();
        cudaDeviceSynchronize();
        return 0;
    }'''

data = {
    "lang": "cuda",
    "filterAnsi": "true",
    "execute": "true",
    "labels": "true",
    "libraryCode": "true",
    "directives": "true",
    "commentOnly": "true",
    "compiler": "nvcc129",
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
