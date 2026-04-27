#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <stdlib.h>
#include <time.h>

#include <cstring>

#define BDIMX 32
#define BDIMY 32

__global__ void setRowReadColumn(int * output)
{
    __shared__ int smem[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    smem[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    //load
    output[idx] = smem[threadIdx.x][threadIdx.y];
}

__global__ void setColumnReadRow(int * output)
{
    __shared__ int smem[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    smem[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();

    //load
    output[idx] = smem[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRow(int * output)
{
    __shared__ int smem[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    smem[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    //load
    output[idx] = smem[threadIdx.y][threadIdx.x];
}

int main(int argc, char ** argv)
{
    int memconfig = 0;
    if (argc > 1)
    {
        memconfig = atoi(argv[1]);
    }

    if (memconfig == 1) {
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    }
    else {
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    }

    cudaSharedMemConfig config;
    cudaDeviceGetSharedMemConfig(&config);
    printf ("With Bank Mode %s ", config == 1 ? "4-Byte" : "8-Byte");
    printf ("\n");

    //Array size 2048
    int nx = BDIMX;
    int ny = BDIMY;

    int total_size = nx * ny * sizeof(int);

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    int *d_C;
    cudaMalloc((int **)&d_C, total_size);
    int * gpuRef = (int *)malloc(total_size);

    cudaMemset(d_C, 0, total_size);

    setColumnReadRow<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(gpuRef, d_C, total_size, cudaMemcpyDeviceToHost);

    for (int i=0; i < total_size; i++) {
        printf("%d ", gpuRef[i]);
    }
    printf("\n");

    cudaMemset(d_C, 0, total_size);
    setRowReadColumn<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(gpuRef, d_C, total_size, cudaMemcpyDeviceToHost);

    for (int i=0; i < total_size; i++) {
        printf("%d ", gpuRef[i]);
    }
    printf("\n");

    cudaMemset(d_C, 0, total_size);
    setRowReadRow<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(gpuRef, d_C, total_size, cudaMemcpyDeviceToHost);

    for (int i=0; i < total_size; i++) {
        printf("%d ", gpuRef[i]);
    }
    printf("\n");

    cudaFree(d_C);
    free(gpuRef);
    
    cudaDeviceReset();
    return 0;
}