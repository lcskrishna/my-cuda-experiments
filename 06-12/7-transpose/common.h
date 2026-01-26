#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

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

// initialize the array.
void initialize(int * input, const int array_size, INIT_PARAM param=INIT_ONE_TO_TEN);

// matrix transpose on cpu.
void matrix_transpose_cpu(int * mat, int * transpose, int nx, int ny);

// compare two arrays provided.
bool compare_arrays(int * a, int * b, int size);

// calculate the elapsed time.
void print_time_using_host_clock(clock_t start, clock_t end);
