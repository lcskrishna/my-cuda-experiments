#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cstring>


enum INIT_PARAM
{
    INIT_ZERO,
    INIT_RANDOM,
    INIT_ONE,
    INIT_ONE_TO_TEN,
};


inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

#define GPU_ERROR_CHECK(ans) \
{ \
    gpuAssert((ans), __FILE__, __LINE__); \
}

void initialize_array(int * array, int size, INIT_PARAM init_type);

bool compare_arrays(int * a, int * b, int size);
