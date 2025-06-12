#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void unique_gid_2d(int * input)
{
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int row_offset = gridDim.x * blockDim.x * blockIdx.y;
    int gid = row_offset + block_offset + tid;

    printf ("blockIdx.x=%d, blockIdx.y=%d, threadIdx.x=%d, gid = %d, value=%d \n", blockIdx.x, blockIdx.y, threadIdx.x, gid, input[gid]);
}

__global__ void unique_gid_2d_2blocks(int * input)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int num_threads_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_threads_block;
    int num_threads_in_block_row = num_threads_block * gridDim.x ;
    int row_offset = num_threads_in_block_row * blockIdx.y;
    int gid = row_offset + block_offset + tid;

    printf ("blockIdx.x=%d, blockIdx.y=%d, threadIdx.x=%d, threadIdx.y=%d, gid = %d, value=%d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, gid, input[gid]);
}

int main()
{
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33, 87, 45, 23, 12, 342, 56, 44, 99};

    for (int i=0;i < array_size; i++) {
        std::cout << h_data[i] << ", ";
    }
    std::cout << std::endl;

    int *d_data;
    cudaMalloc((void **)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(2,2);
    dim3 grid(2,2);

    unique_gid_2d_2blocks<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    
    return 0;
}
