#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// 1 block of threads --> 8 values, grid = 1
__global__ void unique_idx_calc_threadIdx(int * input)
{
    int tid = threadIdx.x;
    printf("threadIdx : %d, value : %d \n", tid, input[tid]);
}

// 4 blocks, each block - 4 threads.
__global__ void unique_gid_calculation(int * input)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;

    printf("blockIdx.x : %d, threadIdx.x : %d, blockDim.x : %d, gridDim.x: %d, value : %d \n", blockIdx.x, tid, blockDim.x , gridDim.x, input[gid]);
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

    dim3 block(4);
    dim3 grid(4);

    //unique_idx_calc_threadIdx<<<grid, block>>>(d_data);
    unique_gid_calculation<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
