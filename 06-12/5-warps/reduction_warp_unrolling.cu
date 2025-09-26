#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "common.h"

__global__ void reduction_warp_unrolling(int * input, int * temp, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int idx = block_offset + tid;
    int *inp_block = input + block_offset;

    for(int offset = blockDim.x / 2; offset >= 64; offset = offset / 2)
    {
        if (tid < offset) {
            inp_block[tid] += inp_block[tid + offset];
        }

        __syncthreads();
    }

    if (tid < 32) { // warpsize.
        volatile int * vol_mem = inp_block;
        vol_mem[tid] += vol_mem[tid + 32];
        vol_mem[tid] += vol_mem[tid + 16];
        vol_mem[tid] += vol_mem[tid + 8];
        vol_mem[tid] += vol_mem[tid + 4];
        vol_mem[tid] += vol_mem[tid + 2];
        vol_mem[tid] += vol_mem[tid + 1];
    }

    if (tid == 0) {
        temp[blockIdx.x] = inp_block[0];
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
    dim3 grid((size/block.x) / 2);  // here 2 is unrolling factor.
    
    std::cout << " Kernel launch parameters = grid.x=" << grid.x << " block.x=" << block.x << std::endl;

    int temp_array_byte_size = sizeof(int) * grid.x;
    h_ref = (int *) malloc(temp_array_byte_size);
    
    int * d_input;
    int * d_temp;
    gpuErrChk(cudaMalloc((void**) &d_input, byte_size));
    gpuErrChk(cudaMalloc((void**) &d_temp, temp_array_byte_size));

    gpuErrChk(cudaMemset(d_temp, 0, temp_array_byte_size));


    gpuErrChk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));
    

    reduction_warp_unrolling<<<grid, block>>>(d_input, d_temp, size); 

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
