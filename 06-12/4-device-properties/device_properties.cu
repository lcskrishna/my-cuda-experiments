#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


void query_device()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "ERROR: No CUDA supported device found." << std::endl;
    }
    
    int devNo = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devNo);
    
    printf ("Device %d: %s\n", devNo, prop.name);
    printf ("   Number of Multiprocessors:   %d \n", prop.multiProcessorCount);
    printf ("   Clock rate:   %d \n", prop.clockRate);
    printf ("   Compute Capability:   %d.%d \n", prop.major, prop.minor);
    printf ("   Total amount of global memory:   %4.2f KB\n", prop.totalGlobalMem / 1024.0);
    printf ("   Amount of constant memory:   %4.2f KB\n", prop.totalConstMem / 1024.0);
    
}

int main()
{
    query_device();
    return 0;
}
