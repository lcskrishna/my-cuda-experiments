#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void initialize(int * input, const int array_size);

int reduction_cpu(int * input, const int size);

void compare_results(int gpu_result, int cpu_result);
