#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void code_without_divergence()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a; float b;
    a = 0; b=0;
    
    int warp_id = gid / 32;
    if (warp_id % 2 == 0) {
        a = 100;
        b = 50;
    }
    else {
        a = 200;
        b = 75;
    }
}

__global__ void code_with_divergence()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a, b;
    a = 0; b=0;
    
    if (gid % 2 == 0) {
        a = 100;
        b = 50;
    }
    else {
        a = 200;
        b = 75;
    }
}

int main()
{
    std::cout << " ---- Warp divergence example ----" << std::endl;

    int size = 1 << 22;
    dim3 block_size(128);
    dim3 grid_size((size + block_size.x - 1)/block_size.x);

    code_without_divergence<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    code_with_divergence<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    
    return 0;
}
