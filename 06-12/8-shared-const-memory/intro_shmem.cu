#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#include "common.h"

#define SHARED_ARRAY_SIZE 128

__global__ void smem_static_test(int * input, int * output, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockDim.x * blockIdx.x;
    int gid = tid + block_offset;

    __shared__ int s_data[SHARED_ARRAY_SIZE];

    if (gid < size) {
        s_data[tid] = input[gid];
        output[gid] = s_data[tid];
    }
    
}

__global__ void smem_dynamic_test(int * input, int * output, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockDim.x * blockIdx.x;
    int gid = tid + block_offset;

    extern __shared__ int s_data[];

    if (gid < size) {
        s_data[tid] = input[gid];
        output[gid] = s_data[tid];
    }
    
}

int main(int argc, char ** argv)
{
    int size = 1 << 22;
    int block_size = SHARED_ARRAY_SIZE;
    int dynamic = false;

    if (argc > 1) {
        dynamic = atoi(argv[1]);
    }

    size_t num_bytes = size * sizeof(int);
    int * h_input, *h_ref, *d_in, *d_out;
    
    //allocate host memory.
    h_input = (int *)malloc(num_bytes);
    h_ref = (int *)malloc(num_bytes);
    initialize_array(h_input, size, INIT_ONE_TO_TEN); 

    // allocate device.
    cudaMalloc((int **)&d_in, num_bytes);
    cudaMalloc((int **)&d_out, num_bytes);

    dim3 block(block_size);
    dim3 grid( (size / block.x) + 1);
    
    // kernel launch.
    cudaMemcpy(d_in, h_input, num_bytes, cudaMemcpyHostToDevice);
    
    if (!dynamic) {
        std::cout << " Static smem kernel. " << std::endl;
        smem_static_test <<< grid, block >>>(d_in, d_out, size);
    }
    else {
        std::cout << " Dynamic smem kernel. " << std::endl;
        smem_dynamic_test<<<grid, block, sizeof(int *) * SHARED_ARRAY_SIZE >>>(d_in, d_out, size);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_ref, d_out, num_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    free(h_input);
    free(h_ref);
    
    cudaDeviceReset();
    return 1;
}
