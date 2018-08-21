#include <stdio.h>
#include <iostream>

#include "cuda_runtime.h"

//Kernel code.
__global__ void square(float * d_in, float * d_out)
{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f;
}


int main()
{
	const int ARRAY_SIZE = 4;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	//input array on host.
	float h_in[ARRAY_SIZE];
	int i;
	for (i=0;i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
		printf("%f\n", h_in[i]);
	}

	//output array of host.
	float h_out[ARRAY_SIZE];
	
	//Declare GPU memory pointers.
	float * d_in;
	float * d_out;

	//Allocate GPU memory.
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);
	
	//transfer the array to GPU.
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	square<<<1,ARRAY_SIZE>>>(d_in, d_out);

	//Copy back to host.
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	//Print the results.
	for (int i=0; i < ARRAY_SIZE; i++) {
		printf("%f\n", h_out[i]);
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
