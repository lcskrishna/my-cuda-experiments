#include <iostream>
#include <stdio.h>

#include "cuda_runtime.h"

__global__ void cubic(const float * d_in, float * d_out)
{
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f * f;
}

int main()
{
    const int ARRAY_SIZE=10;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    for(int i=0;i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }

    float h_out[ARRAY_SIZE];
    
    //Declare GPU memory.
    float * d_in;
    float * d_out;
    
    cudaMalloc((void **)&d_in, ARRAY_BYTES);
    cudaMalloc((void **)&d_out, ARRAY_BYTES);

    //Tranfer array to GPU.
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    cubic<<<1, ARRAY_SIZE>>>(d_in, d_out);

    //copy bakc the results back to host.
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    //Output the results.
    for(int i=0; i < ARRAY_SIZE; i++) {
        printf("%f\n", h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
