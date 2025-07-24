#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_details_of_warps()
{
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int row_offset = blockIdx.y * gridDim.x * blockDim.x;

    int gid = tid + block_offset + row_offset;
    int warp_id = threadIdx.x / 32;
    
    int gbid = blockIdx.y * gridDim.x + blockIdx.x;
    printf("tid: %d, bid.x: %d, bid.y: %d, gid: %d, warp_id: %d, gbid: %d \n", tid, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

int main()
{
    dim3 block(42);
    dim3 grid(2, 2);

    print_details_of_warps<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    
    return 0;
}
