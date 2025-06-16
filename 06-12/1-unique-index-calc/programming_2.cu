#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void print_details(int * input)
{
    int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int num_threads_block = blockDim.x * blockDim.y * blockDim.z;
    int block_offset = num_threads_block * blockIdx.x;
    int num_threads_per_row = num_threads_block * gridDim.x;
    int row_offset = num_threads_per_row * blockIdx.y;
    int num_threads_per_xy = num_threads_block * gridDim.x * gridDim.y;
    int z_offset = num_threads_per_xy * blockIdx.z;
    
    int gid = tid + block_offset + row_offset + z_offset;
    
    printf("tid: %d, gid: %d, value: %d \n", tid, gid, input[gid]);
    
}


int main()
{
    int size = 64;
    int size_in_bytes = size * sizeof(int);
    
    int * h_input;
    h_input = (int*) malloc(size_in_bytes);
    
    // initilize
    time_t t;
    srand((unsigned)time(&t));
    
    for(int i=0; i < size; i++) {
        h_input[i] = (int)(rand() & 0xff);
    }

    for (int i=0; i < size; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;
    

    // device.
    int * d_input;
    cudaMalloc((void**)&d_input, size_in_bytes);
    cudaMemcpy(d_input, h_input, size_in_bytes, cudaMemcpyHostToDevice);
    
    dim3 block(2,2,2);
    dim3 grid(2,2,2);

    print_details<<<grid, block>>>(d_input);

    cudaDeviceSynchronize();
    cudaFree(d_input);
    free(h_input);

    cudaDeviceReset();
    return 0;

}
