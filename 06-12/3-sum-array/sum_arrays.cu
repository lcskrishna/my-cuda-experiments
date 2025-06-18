#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "cuda_common.h"
#include <stdio.h>
#include <iostream>

#include <stdlib.h>
#include <time.h>

#include <cstring>

__global__ void sum_array(int *a, int * b, int *c, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockDim.x * blockIdx.x;
    int gid = tid + block_offset;

    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}


void compare_arrays(int *a, int *b, int size)
{
    for (int i=0; i < size; i++) {
        if (a[i] != b[i]) {
            std::cout << "Arrays are different" << std::endl;
            return;
        }
    }
    std::cout << "Arrays are same" << std::endl;
}

void sum_array_cpu(int *a, int * b, int * result, int size)
{
    for (int i=0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

int main()
{
    int size = 10000;
    int block_size = 128;
    int byte_size = size * sizeof(int);
    
    int * h_a, *h_b;
    int * gpu_results, *h_c;
    
    h_a = (int*)malloc(byte_size);
    h_b = (int*)malloc(byte_size);
    gpu_results = (int*)malloc(byte_size);
    h_c = (int *)malloc(byte_size);
   
    time_t t;
    srand((unsigned)time(&t));
    for (int i=0; i < size; i++) {
        h_a[i] = (int)(rand() & 0xff);
    }
    
    for(int i=0; i < size; i++) {
        h_b[i] = (int) (rand() & 0xff);
    }

    memset(gpu_results, 0, byte_size);

    sum_array_cpu(h_a, h_b, h_c, size);
    
    int * d_a, *d_b;
    int * d_c;
    cudaMalloc((void**)&d_a, byte_size);
    cudaMalloc((void**)&d_b, byte_size);
    cudaMalloc((void**)&d_c, byte_size);

    cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((size/block.x) + 1);

    sum_array<<<grid, block>>>(d_a, d_b, d_c, size);

    cudaDeviceSynchronize();
    cudaMemcpy(gpu_results, d_c, byte_size, cudaMemcpyDeviceToHost);

    compare_arrays(gpu_results, h_c, size);
   
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    
    free(gpu_results);
    free(h_b);
    free(h_a); 
    free(h_c);

    cudaDeviceReset();
    return 0;
}
