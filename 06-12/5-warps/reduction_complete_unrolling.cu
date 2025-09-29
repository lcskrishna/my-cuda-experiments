#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "common.h"

__global__ void reduction_complete_unrolling(int * input, int * temp, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockDim.x * blockIdx.x;
    int idx = tid + block_offset;
    
    int * input_block = input + block_offset;
    
    if (blockDim.x == 4096 && tid < 2048) {
        input_block[tid] += input_block[tid + 2048];
    }
    __syncthreads();

    if (blockDim.x == 2048 && tid < 1024) {
        input_block[tid] += input_block[tid + 1024];
    }
    __syncthreads();

    if (blockDim.x == 1024 && tid < 512) {
        input_block[tid] += input_block[tid + 512];
    }
    __syncthreads();

    if (blockDim.x == 512 && tid < 256) {
        input_block[tid] += input_block[tid + 256];
    }
    __syncthreads();

    if (blockDim.x == 256 && tid < 128) {
        input_block[tid] += input_block[tid + 128];
    }
    __syncthreads();

    if (blockDim.x == 128 && tid < 64) {
        input_block[tid] += input_block[tid + 64];
    }
    __syncthreads();


    // the above removed the for loop in previous warp level parallelism.
    if (tid < 32) {
        volatile int * shmem = input_block;
        shmem[tid] += shmem[tid + 32];
        shmem[tid] += shmem[tid + 16];
        shmem[tid] += shmem[tid + 8];
        shmem[tid] += shmem[tid + 4];
        shmem[tid] += shmem[tid + 2];
        shmem[tid] += shmem[tid + 1];
    }

    if (tid == 0) {
        temp[blockIdx.x] = input_block[0];
    }
}

int main(int argc, char** argv)
{

    std::cout << "Running warp unroll reduction kernel." << std::endl;
    int size = 1 << 27; // 128MB 
    int byte_size = size * sizeof(int);
    int block_size = 128;

    int * h_input;
    int * h_ref;

    h_input = (int *)malloc(byte_size);
    initialize(h_input, size);
    
    int cpu_result = reduction_cpu(h_input, size);

    dim3 block(block_size);
    dim3 grid(size/block.x);  // here 2 is unrolling factor.
    
    std::cout << " Kernel launch parameters = grid.x=" << grid.x << " block.x=" << block.x << std::endl;

    int temp_array_byte_size = sizeof(int) * grid.x;
    h_ref = (int *) malloc(temp_array_byte_size);
    
    int * d_input;
    int * d_temp;
    gpuErrChk(cudaMalloc((void**) &d_input, byte_size));
    gpuErrChk(cudaMalloc((void**) &d_temp, temp_array_byte_size));

    gpuErrChk(cudaMemset(d_temp, 0, temp_array_byte_size));


    gpuErrChk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));
    

    reduction_complete_unrolling<<<grid, block>>>(d_input, d_temp, size); 

    cudaDeviceSynchronize();
    cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);
    
    int gpu_result = 0;
    for (int i=0; i < grid.x; i++) {
        gpu_result += h_ref[i];
    }

    // validate
    compare_results(gpu_result, cpu_result);

    cudaFree(d_temp);
    cudaFree(d_input);
    free(h_ref);
    free(h_input);


    cudaDeviceReset();
    return 0;
}
