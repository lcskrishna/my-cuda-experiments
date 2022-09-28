#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// 2 - 2d block of threads --> 4 values in each dimension of x and y , grid = 2
__global__ void unique_gid_calculation2d(int * input)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int num_threads_per_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_threads_per_block;
    int num_threads_in_row = num_threads_per_block * gridDim.x;
    int row_offset = num_threads_in_row * blockIdx.y;

    int gid = tid + block_offset + row_offset;
    
    printf("blockIdx.x : %d, blockIdx.y : %d, threadIdx.x : %d, gid : %d - data : %d \n", blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

int main()
{
    int array_size = 16;
    int array_bite_size = sizeof(int) * array_size;
    int h_data[] = {23, 9, 4, 53, 64, 12, 1, 33, 22, 11, 9, 12, 13, 89, 90, 77};

    for (int i=0; i < array_size; i++) {
        printf("%d ", h_data[i]);
    }
    printf ("\n \n");

    int * d_data;
    cudaMalloc((void **)&d_data, array_bite_size);
    cudaMemcpy(d_data, h_data, array_bite_size, cudaMemcpyHostToDevice);

    dim3 block(2,2);
    dim3 grid(2,2);

    //unique_idx_calc_threadIdx<<<grid, block>>>(d_data);
    unique_gid_calculation2d<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
