#include "common.h"

__global__ void copy_row(int * mat, int * transpose, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        transpose[iy * nx + ix] = mat[iy * nx + ix];
    }
}

__global__ void copy_column(int * mat, int * transpose, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        transpose[ix * ny + iy] = mat[ix * ny + iy];
    }
}

__global__ void transpose_read_row_write_column(int * mat, int * transpose, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        transpose[ix * ny + iy] = mat[iy * nx + ix];
    }
}

__global__ void transpose_read_column_write_row(int * mat, int * transpose, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        transpose[iy * nx + ix] = mat[ix * ny + iy];
    }
}

__global__ void transpose_unroll4_row(int * mat, int * transpose, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    int ti = iy * nx + ix;
    int to = ix * ny + iy;

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        transpose[to]                        = mat[ti];
        transpose[to + ny * blockDim.x]      = mat[ti + blockDim.x];
        transpose[to + ny * 2 * blockDim.x]  = mat[ti + 2 * blockDim.x];
        transpose[to + ny * 3 * blockDim.x]  = mat[ti + 3 * blockDim.x];
    }
}

__global__ void transpose_unroll4_col(int * mat, int * transpose, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    int ti = iy * nx + ix;
    int to = ix * ny + iy;

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        transpose[ti]                        = mat[to];
        transpose[ti + blockDim.x]      = mat[to + blockDim.x * ny];
        transpose[ti + 2 * blockDim.x]  = mat[to + 2 * blockDim.x * ny];
        transpose[ti + 3 * blockDim.x]  = mat[to + 3 * blockDim.x * ny];
    }
}

__global__ void transpose_diagonal_row(int * mat, int * transpose, int nx, int ny)
{
	int blk_x = blockIdx.x;
	int blk_y = (blockIdx.x + blockIdx.y) % gridDim.x;

	int ix = blockIdx.x * blk_x + threadIdx.x;
	int iy = blockIdx.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}
}

int main(int argc, char ** argv)
{
    if (argc < 2) {
        printf ("Usage: mat_transpose <algo> ranging from 0 to 3 \n");
        exit(1);
    } 

    int kernel_num = atoi(argv[1]);
    printf ("Kernel number selected is : %d \n", kernel_num);

    int nx = 1024;
    int ny = 1024;
    int block_x = 128;
    int block_y = 8;
   
    int total_size = nx * ny;
    int total_bytes = total_size * sizeof(int);

    printf ("Matrix transpose for %d x %d matrix with block_size %d x %d \n", nx, ny, block_x, block_y); 

    int * h_mat_array = (int *)malloc(total_bytes);
    int * h_transpose = (int *)malloc(total_bytes);
    int * h_ref = (int *)malloc(total_bytes);

    // initialize.
    initialize(h_mat_array, total_size, INIT_ONE_TO_TEN);

    // matrix transpose in cpu.
    matrix_transpose_cpu(h_mat_array, h_transpose, nx, ny);

    //std::cout << "Original Matrix is: " << std::endl;
    //for (int i=0; i < total_size; i++) {
    //    std::cout << h_mat_array[i] << std::endl;
    //}

    //std::cout << "Transpose matrix is: " << std::endl;
    //for (int i=0; i < total_size; i++) {
    //    std::cout << h_transpose[i] << std::endl;
    //}

    int * d_array, * d_transpose;
    GPU_ERROR_CHECK(cudaMalloc((void**)&d_array, total_bytes));
    GPU_ERROR_CHECK(cudaMalloc((void**)&d_transpose, total_bytes));

    GPU_ERROR_CHECK(cudaMemcpy(d_array, h_mat_array, total_bytes, cudaMemcpyHostToDevice));

    dim3 blocks(block_x, block_y);
    dim3 grid(nx/block_x, ny / block_y);

    void(*kernel)(int *, int *, int , int);
    char * kernel_name;

    switch (kernel_num)
    {
        case 0:
            kernel = &copy_row;
            kernel_name = "Copy Row";
            break;

        case 1:
            kernel = &copy_column;
            kernel_name = "Copy Column kernel";
            break;

        case 2:
            kernel = &transpose_read_row_write_column;
            kernel_name = "Tranpose read row & write column";
            break;

        case 3:
            kernel = &transpose_read_column_write_row;
            kernel_name = "Read column write row ";
            break;

        case 4:
            kernel = &transpose_unroll4_row;
            kernel_name = "Unroll 4 row ";
            break;

        case 5:
            kernel = &transpose_unroll4_col;
            kernel_name = "Unroll 4 col ";
            break;

        case 6:
            kernel = &transpose_diagonal_row;
            kernel_name = "Diagonal row ";
            break;
        default:
            kernel = &copy_row;
            kernel_name = "Copy Row";
    }

    std::cout << "launching kernel -> " << kernel_name << std::endl;
    
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    
    kernel <<< grid, blocks >>>(d_array, d_transpose, nx, ny);

    cudaDeviceSynchronize();
    gpu_end = clock();
    print_time_using_host_clock(gpu_start, gpu_end);


    // copy memory back to cpu.
    GPU_ERROR_CHECK(cudaMemcpy(h_ref, d_transpose, total_bytes, cudaMemcpyDeviceToHost));
    
    bool issame = compare_arrays(h_ref, h_transpose, total_size);
    if (issame) {
        std::cout << "OK: Arrays are same." << std::endl;
    } else {
        std::cout << "ERROR: Arrays are not same" << std::endl;
    }

    cudaDeviceReset();

    cudaFree(d_array);
    cudaFree(d_transpose);

    free(h_mat_array);
    free(h_transpose);
    free(h_ref);

    return 0;
}
