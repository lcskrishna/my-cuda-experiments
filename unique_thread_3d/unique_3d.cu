#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void calculate_unique_3d_idx(int * input, int size)
{
    int tid = (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int block_id = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    int global_index = block_id * blockDim.x * blockDim.y * blockDim.z + tid;

    printf ("tid: %d, block_id : %d, global_index : %d, value: %d \n", tid, block_id, global_index, input[global_index]);
}

int main()
{
    int size = 64;
    int byte_size = sizeof(int) * size;
    
    int * h_data;
    h_data = (int *) malloc(byte_size);
    
    time_t t;
    srand((unsigned) time(&t));
    for (int i =0; i < size; i++) {
        h_data[i] = (int) (rand() && 0xff);
    }

    int * d_data;
    cudaMalloc((void **)&d_data, byte_size);
    cudaMemcpy(d_data, h_data, byte_size, cudaMemcpyHostToDevice);
    

    int nx, ny, nz;
    nx = 4; ny = 4; nz = 4;
    
    dim3 block(2, 2, 2);
    dim3 grid(nx/block.x, ny/block.y, nz/block.z);

    calculate_unique_3d_idx<<<grid, block>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    free(h_data);
    
    cudaDeviceReset();
    return 0;
}
