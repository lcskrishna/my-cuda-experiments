#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda()
{
    printf("Hello CUDA world : \n");
}

int main()
{
    //hello_cuda <<<1,10>>>();
    // 
    //dim3 block(4, 1, 1);
    //dim3 grid(8, 1, 1);
    int nx, ny;
    nx = 16;
    ny = 4;

    dim3 block(8, 2);
    dim3 grid(nx/block.x , ny/block.y);

    hello_cuda <<< grid, block >>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return 0;
}

