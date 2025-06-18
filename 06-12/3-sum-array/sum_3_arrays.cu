#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "cuda_common.h"
#include <stdio.h>
#include <iostream>

#include <stdlib.h>
#include <time.h>

#include <cstring>

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d \n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void sum_three_array(int *a, int * b, int *c, int *d, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockDim.x * blockIdx.x;
    int gid = tid + block_offset;

    if (gid < size) {
        d[gid] = a[gid] + b[gid] + c[gid];
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

void sum_array_cpu(int *a, int * b, int *c,  int * result, int size)
{
    for (int i=0; i < size; i++) {
        result[i] = a[i] + b[i] + c[i];
    }
}

int main()
{
    int size = 1 << 25;
    int block_size = 128;
    int byte_size = size * sizeof(int);
    
    int * h_a, *h_b, *h_c;
    int * gpu_results, *h_d;
    
    h_a = (int*)malloc(byte_size);
    h_b = (int*)malloc(byte_size);
    gpu_results = (int*)malloc(byte_size);
    h_c = (int *)malloc(byte_size);
    h_d = (int *)malloc(byte_size);
   
    time_t t;
    srand((unsigned)time(&t));
    for (int i=0; i < size; i++) {
        h_a[i] = (int)(rand() & 0xff);
    }
    
    for(int i=0; i < size; i++) {
        h_b[i] = (int) (rand() & 0xff);
    }

    for (int i=0; i < size; i++) {
        h_c[i] = (int)(rand() & 0xff);
    }

    memset(gpu_results, 0, byte_size);

    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_c, h_d, size);
    cpu_end = clock();
    
    int * d_a, *d_b;
    int * d_c, *d_d;
    gpuErrCheck(cudaMalloc((void**)&d_a, byte_size));
    gpuErrCheck(cudaMalloc((void**)&d_b, byte_size));
    gpuErrCheck(cudaMalloc((void**)&d_c, byte_size));
    gpuErrCheck(cudaMalloc((void**)&d_d, byte_size));

    clock_t htod_start, htod_end;
    htod_start = clock();
    gpuErrCheck(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(d_c, h_c, byte_size, cudaMemcpyHostToDevice));
    htod_end = clock();

    dim3 block(block_size);
    dim3 grid((size/block.x) + 1);

    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sum_three_array<<<grid, block>>>(d_a, d_b, d_c, d_d, size);
    gpu_end = clock();

    cudaDeviceSynchronize();
    clock_t hdtoh_start, hdtoh_end;
    hdtoh_start = clock();
    cudaMemcpy(gpu_results, d_d, byte_size, cudaMemcpyDeviceToHost);
    hdtoh_end = clock();

    compare_arrays(gpu_results, h_d, size);

    std::cout << "Sum array CPU execution time: " << (double)(((double)(cpu_end - cpu_start)/CLOCKS_PER_SEC)) << std::endl;
    std::cout << "Sum array GPU execution time: " << (double)(((double)(gpu_end - gpu_start)/CLOCKS_PER_SEC)) << std::endl;
    std::cout << "Host to Device Transfer time: " << (double)(((double)(htod_end - htod_start)/CLOCKS_PER_SEC)) << std::endl;
    std::cout << "Device to Host Transfer time: " << (double)(((double)(hdtoh_end - hdtoh_start)/CLOCKS_PER_SEC)) << std::endl;
   
    cudaFree(d_d);
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    
    free(gpu_results);
    free(h_b);
    free(h_a); 
    free(h_c);
    free(h_d);

    cudaDeviceReset();
    return 0;
}
