#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_details_of_wraps()
{
    int gid = (blockIdx.y * gridDim.x * blockDim.x) + (blockDim.x * blockIdx.x) + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int gbid = blockIdx.y * gridDim.x + blockIdx.x;

    printf("tid: %d, bid.x : %d, bid.y : %d, gid: %d, warp_id : %d, gbid : %d \n", threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

int main(int argc , char ** argv)
{
    dim3 block(42);
    dim3 grid(2, 2);
    
    print_details_of_wraps<<<grid, block>>>();
    cudaDeviceSynchronize();
    
    cudaDeviceReset();
    return 0;
}
