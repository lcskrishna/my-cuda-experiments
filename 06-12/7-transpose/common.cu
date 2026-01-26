#include "common.h"

void initialize(int * input, const int array_size, INIT_PARAM param)
{
    if (param == INIT_ONE) {
        for (int i=0; i < array_size; i++) {
            input[i] = 1;
        }
    }
    else if (param == INIT_ONE_TO_TEN) {
        time_t t;
        srand((unsigned)time(&t));
        for (int i=0; i < array_size; i++) {
            input[i] = (int)(rand() & 0xFF) % 10;
        }
    }
    else if (param == INIT_RANDOM) {
        time_t t;
        srand((unsigned)time(&t));
        for (int i=0; i < array_size; i++) {
            input[i] = (int)(rand() & 0xFF);
        }
    } else {
        std::cout << "ERROR: Unsupported parameter passed = " << param << std::endl;
    }
}

void matrix_transpose_cpu(int * mat, int * transpose, int nx, int ny)
{
    for (int iy=0; iy < ny; iy++) {
        for (int ix=0; ix < nx; ix++) {
            transpose[ix * ny + iy] = mat[iy * nx + ix];
        }
    }
}

bool compare_arrays(int * a, int * b, int size)
{
    bool issame = true;
    for (int i=0; i < size; i ++) {
        if (a[i] != b[i]) {
            issame = false;
            break;
            return issame;
        }
    }
    return issame;
}

void print_time_using_host_clock(clock_t start, clock_t end)
{
    double exec_time = (double)((double)(end - start)/ CLOCKS_PER_SEC);
    printf ("GPU Execution time: %4.6f \n", exec_time);
}
