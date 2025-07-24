#include "common.h"
#include "time.h"
#include <cstring>

void initialize(int * input, const int array_size)
{
    time_t t;
    srand((unsigned)time(&t));
    for(int i=0; i < array_size; i++) {
        input[i] = (int)(rand() & 0xff);
    }
}

int reduction_cpu(int * input, const int size)
{
    int sum = 0;
    for (int i=0; i < size; i++) {
        sum += input[i];
    }

    return sum;
}

void compare_results(int gpu_result, int cpu_result)
{
    std::cout << "GPU Result=" << gpu_result << " , CPU Result=" << cpu_result << std::endl;
    if (gpu_result == cpu_result) {
        std::cout << "Arrays are same." << std::endl;
    } else {
        std::cout << "Results don't match between CPU and GPU" << std::endl;
    }
}
