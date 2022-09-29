#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

/* 
1. three arrays of size 2 ^ 22 randomly initialized.
2. cpu comparision for three arrays sum
3. gpu kernel to sum three arrays. 
4. cuda error mechanism.
5. grid is 1D.
6. check with block size - 64, 128, 256, 512. 

*/

#define CHECK_ERROR(value) { check_cuda_error((value), __FILE__, __LINE__); }
inline void check_cuda_error(cudaError_t error, const char * file, int line, bool abort = true)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d \n", cudaGetErrorString(error), file, line);
        if (abort) {
            exit(error);
        }
    }
}

// GPU device kernel.
__global__ void sum_three_arrays_gpu(int * a, int * b, int * c, int * d, int size)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < size) {
        d[global_id] = a[global_id] + b[global_id] + c[global_id];
    }
}

// CPU implementation. 
void sum_three_arrays_cpu(int * a, int * b, int * c, int * d, int size)
{
    for (int i=0; i < size; i++) {
        d[i] = a[i] + b[i] + c[i];
    }
}

bool compare_two_arrays(int * a, int * b, int size)
{
    bool same = true;
    for (int i=0; i < size; i++) {
        if (a[i] != b[i]) {
            same = false;
            break;
        }
    }
    return same;
}

int main(int argc, char * argv[])
{

    int size = 2 << 22;
    int byte_size = size * sizeof(int);
    int block_size = 256;
    std::cout << "Experiment: Block size used is:  " << block_size << std::endl;

    //Create Host arrays.
    int * h_a, *h_b, *h_c, *gpu_results, *cpu_results;
    h_a = (int *)malloc(byte_size);
    h_b = (int *)malloc(byte_size);
    h_c = (int *)malloc(byte_size);
    gpu_results = (int *)malloc(byte_size);
    cpu_results = (int *)malloc(byte_size);

    time_t t;
    srand((unsigned) time(&t));
    
    //Initalize random values for arrays.
    //initialize array a
    for(int i=0; i < size; i++) {
        h_a[i] = (int)(rand() && 0xff);
    }

    //initailize array b
    for(int i=0; i < size; i++) {
        h_a[i] = (int)(rand() && 0xff);
    }

    //initialize array c.
    for(int i=0; i < size; i++) {
        h_a[i] = (int)(rand() && 0xff);
    }

    memset(gpu_results, 0, byte_size);
    memset(cpu_results, 0, byte_size);

    //CPU Results.
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_three_arrays_cpu(h_a, h_b, h_c, cpu_results, size);
    cpu_end = clock();

    // Allocate device memory.
    int * d_a, *d_b, *d_c, *d_results;
    CHECK_ERROR(cudaMalloc((int **)&d_a, byte_size));
    CHECK_ERROR(cudaMalloc((int **)&d_b, byte_size));
    CHECK_ERROR(cudaMalloc((int **)&d_c, byte_size));
    CHECK_ERROR(cudaMalloc((int **)&d_results, byte_size));

    // Move memory host to device.

    clock_t htod_start, htod_end;
    htod_start = clock();
    CHECK_ERROR(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_c, h_c, byte_size, cudaMemcpyHostToDevice));
    htod_end = clock();
    
    // Grid and block Size for device execution.
    dim3 block(block_size);
    dim3 grid((size/block.x) + 1);
    
    // Device execution of summation.
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sum_three_arrays_gpu<<<grid, block>>>(d_a, d_b, d_c, d_results, size);
    cudaDeviceSynchronize();
    gpu_end = clock();
    
    //Device To Host.
    clock_t dtoh_start, dtoh_end;
    dtoh_start = clock();
    CHECK_ERROR(cudaMemcpy(gpu_results, d_results, byte_size, cudaMemcpyDeviceToHost));
    dtoh_end = clock();
    
    // compare the results.
    auto result = compare_two_arrays(gpu_results, cpu_results, size);
    if (result) {
        printf("Both the CPU and GPU results match. \n");
    } else {
        printf("Mismatch in CPU and GPU results.\n");
    }
    
    // Print execution times.
    printf("Sum array CPU execution time      : %4.6f \n" , (double)((double)(cpu_end - cpu_start)/ CLOCKS_PER_SEC));
    printf("H to D mem transfer time          : %4.6f \n" , (double)((double)(htod_end - htod_start)/ CLOCKS_PER_SEC));
    printf("Sum array GPU execution time      : %4.6f \n" , (double)((double)(gpu_end - gpu_start)/ CLOCKS_PER_SEC));
    printf("D to H mem transfer time          : %4.6f \n" , (double)((double)(dtoh_end - dtoh_start)/ CLOCKS_PER_SEC));
    printf("Sum array GPU total execution time: %4.6f \n" , (double)((double)(dtoh_end - htod_start)/ CLOCKS_PER_SEC));

    CHECK_ERROR(cudaFree(d_results));
    CHECK_ERROR(cudaFree(d_c));
    CHECK_ERROR(cudaFree(d_b));
    CHECK_ERROR(cudaFree(d_a));
    
    free(gpu_results);
    free(cpu_results);
    free(h_a);
    free(h_b);
    free(h_c);

    cudaDeviceReset();
    return 0;
}
