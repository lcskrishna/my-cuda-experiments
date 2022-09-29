#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void query_device()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("No CUDA support available \n");
        exit(1);
    }

    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);

    printf(" Device %d: %s \n", devNo, iProp.name);

    cudaDeviceReset();
}

int main()
{
    query_device();
    return 0;
}
