#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_thread_ids()
{
    printf("threadIdx.x = %d, threadIdx.y: %d, threadIdx.z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
} 


int main()
{
    int nx, ny;
    nx = 16;
    ny = 16;


    dim3 block(8, 8);
    dim3 grid(nx/block.x, ny/block.y);

    print_thread_ids <<< grid, block>>>();
    cudaDeviceReset();

    return 0;
}
