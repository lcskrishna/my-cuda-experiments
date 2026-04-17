nvcc --ptxas-options=-v -o shmem_test intro_shmem.cu common.cu -I common.h
