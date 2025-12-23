#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main(int argc, char ** argv)
{
    int isize = 1 << 25;
    int nbytes = isize * sizeof(float);
    
    float *h_a = (float*)malloc(nbytes);
    //float *h_a;
    //cudaMallocHost((float **)&h_a, nbytes);

    float *d_a;
    cudaMalloc((float **)&d_a, nbytes);
    
    for (int i=0; i < isize; i++)
    {
        h_a[i] = 7;
    }

    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    free(h_a);
    //cudaFreeHost(h_a);

    cudaDeviceReset();
    return 0;
    
}
