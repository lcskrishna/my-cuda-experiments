#include "device_tensor.cuh"

/*
FILL KERNEL
This kernel will fill an allocatoin with a constant value 'val'.
*/

template<int N_DIMS>
__global__ void
kernel_fill_apply_v2_dim1(device_tensor<N_DIMS> x, const float val){
  int tid = threadIdx.x;
  int offset = blockIdx.x * blockDim.x;
  int gid = tid + offset;
  x.at_linear(gid) = val;
}

template<int N_DIMS>
__global__ void
kernel_fill_apply_v2_dim2(device_tensor<N_DIMS> x, const float val){
  int tid = blockDim.x * threadIdx.y + threadIdx.x;
  int num_threads_per_block = blockDim.x * blockDim.y;
  int block_offset = blockIdx.x * num_threads_per_block;
  int num_threads_in_row = num_threads_per_block * gridDim.x;
  int row_offset = num_threads_in_row * blockIdx.y;
  int gid = tid + block_offset + row_offset;
  x.at_linear(gid) = val;
}

template<int N_DIMS>
__global__ void
kernel_fill_apply(device_tensor<N_DIMS> x, const float val){

  size_t i = threadIdx.x;
  while(i < x.get_n_elems()){
    x.at_linear(i) = val;
    i += blockDim.x;
  }

}

//GPU kernel wrapper for fill_apply.
template<int N_DIMS>
void fill_apply(device_tensor<N_DIMS>& x, const float val){
  if (N_DIMS == 2) {
    int rows = x.size[0];
    int cols = x.size[1];
    dim3 block(32, 32);
    dim3 grid(rows/block.x, cols/block.y);
    kernel_fill_apply_v2_dim2<N_DIMS> <<<grid, block>>>(x, val);
  } else if (N_DIMS == 1) {
    int cols = x.size[0];
    dim3 block(32);
    dim3 grid(cols/block.x);
    kernel_fill_apply_v2_dim1<N_DIMS> <<<grid, block>>>(x, val);
  } else {
    kernel_fill_apply<N_DIMS> <<<1, 32>>>(x, val);
  }
}


/*
PPOINTWISE KERNEL
Will loop over elements of tensor x and y and apply an elementwise operation between them.
i.e. out[i][j] = op::op(x[i][j], y[i][j])
*/
template<typename op, int N_DIMS>
__global__ void
kernel_pointwise_apply(device_tensor<N_DIMS> out,
		       const device_tensor<N_DIMS> x, const device_tensor<N_DIMS> y){

  size_t i = threadIdx.x;
  while(i < out.get_n_elems()){
    out.at_linear(i) = op::op(x.at_linear(i), y.at_linear(i));
    i += blockDim.x;
  }

}

template<typename op, int N_DIMS>
__global__ void
kernel_pointwise_apply_v2_dim1(device_tensor<N_DIMS> out,
		       const device_tensor<N_DIMS> x, const device_tensor<N_DIMS> y){

  int tid = threadIdx.x;
  int offset = blockDim.x * blockIdx.x;
  int gid = tid + offset;

  out.at_linear(gid) = op::op(x.at_linear(gid), y.at_linear(gid));
}

template<typename op, int N_DIMS>
__global__ void
kernel_pointwise_apply_v2_dim2(device_tensor<N_DIMS> out,
		       const device_tensor<N_DIMS> x, const device_tensor<N_DIMS> y){

  int tid = blockDim.x * threadIdx.y + threadIdx.x;
  int num_threads_per_block = blockDim.x * blockDim.y;
  int block_offset = blockIdx.x * num_threads_per_block;
  int num_threads_in_row = num_threads_per_block * gridDim.x;
  int row_offset = num_threads_in_row * blockIdx.y;
  int gid = tid + block_offset + row_offset;

  out.at_linear(gid) = op::op(x.at_linear(gid), y.at_linear(gid));
}

//GPU kernel wrapper for pointwise apply
template<typename op, int N_DIMS>
device_tensor<N_DIMS> pointwise_apply(const device_tensor<N_DIMS>& x, const device_tensor<N_DIMS>& y){
  assert( x.get_n_elems() == y.get_n_elems() );
  device_tensor<N_DIMS> out(x.size);
  if (N_DIMS == 2) {
    int rows = x.size[0];
    int cols = x.size[1];
    dim3 block(32, 32);
    dim3 grid(rows/block.x, cols/block.y);
    kernel_pointwise_apply_v2_dim2<op, N_DIMS> <<<grid, block>>>(out, x, y);
    return out;
  } else if (N_DIMS == 1) {
    int cols = x.size[0];
    dim3 block(32);
    dim3 grid(cols/block.x);
    kernel_pointwise_apply_v2_dim1<op, N_DIMS> <<<grid, block>>>(out, x, y);
    return out;
  } else {
    kernel_pointwise_apply<op, N_DIMS> <<<1, 32>>>(out, x, y);
    return out;
  }
}


/*
PPOINTWISE KERNEL
Will loop over elements of tensor x and apply an elementwise operation on it.
i.e. out[i][j] = op::op(x[i][j])
*/
template<typename op, int N_DIMS>
__global__ void
kernel_pointwise_apply(device_tensor<N_DIMS> out,
		       const device_tensor<N_DIMS> x){

  size_t i = threadIdx.x;
  while(i < out.get_n_elems()){
    out.at_linear(i) = op::op(x.at_linear(i));
    i += blockDim.x;
  }

}

template<typename op, int N_DIMS>
__global__ void
kernel_pointwise_apply_v2_dim1(device_tensor<N_DIMS> out,
		       const device_tensor<N_DIMS> x){
    int tid = threadIdx.x;
    int offset = blockDim.x * blockIdx.x;
    int gid = tid + offset;

    out.at_linear(gid) = op::op(x.at_linear(gid));
}

template<typename op, int N_DIMS>
__global__ void
kernel_pointwise_apply_v2_dim2(device_tensor<N_DIMS> out,
		       const device_tensor<N_DIMS> x){
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int num_threads_per_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_threads_per_block;
    int num_threads_in_row = num_threads_per_block * gridDim.x;
    int row_offset = num_threads_in_row * blockIdx.y;
    int gid = tid + block_offset + row_offset;

    out.at_linear(gid) = op::op(x.at_linear(gid));
}

//GPU kernel wrapper for pointwise apply
template<typename op, int N_DIMS>
device_tensor<N_DIMS> pointwise_apply(const device_tensor<N_DIMS>& x)
{
  device_tensor<N_DIMS> out(x.size);
  if (N_DIMS == 2) {
    int rows = x.size[0];
    int cols = x.size[1];
    dim3 block(32, 32);
    dim3 grid(rows/block.x, cols/block.y);
    kernel_pointwise_apply_v2_dim2<op, N_DIMS> <<<grid, block>>>(out, x);
    return out; 
  } else if (N_DIMS == 1) {
    int cols = x.size[0];
    dim3 block(32);
    dim3 grid(cols/block.x);
    kernel_pointwise_apply_v2_dim1<op> <<<grid, block>>>(out, x);
    return out;
  } else {
    kernel_pointwise_apply<op, N_DIMS> <<<1, 32>>>(out, x);
    return out;
  }
}


/* REDUCTION KERNEL
Takes a 2D tensor and reduces its second dimension based on op::op.
i.e. out[i] = init_value
     for j in J:
         out[i] = op::op(out[i], A[i, j])
*/
template<typename op>
__global__ void  reduce_dim_1(device_tensor<1> out,
	     const device_tensor<2> in){

  size_t i = threadIdx.x;
  while(i<in.size[0]){

    float red = op::init();
    for(size_t j=0; j<in.size[1]; j++){
      red = op::op(in.at(i, j), red);
    }

    out.at(i) = red;
    i += blockDim.x;

  }

}

template<typename op>
__global__ void  reduce_dim_1_v2(device_tensor<1> out,
	     const device_tensor<2> in){
  int rowidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowidx < in.size[0]) {
    float red = op::init();
    for (int j = 0; j < in.size[1]; j++) {
        red = op::op(in.at_linear(rowidx * in.size[1] + j), red);
    }
    out.at(rowidx) = red;
  }

}

//GPU kernel wrapper for reduce dim=1
template<typename op>
device_tensor<1> reduce_apply(const device_tensor<2>& x)
{
  device_tensor<1> out({x.size[0]});
  dim3 block(32);
  dim3 grid(x.size[0]/block.x);
#if 0
  reduce_dim_1<op> <<<grid, block>>>(out, x);
#else
  reduce_dim_1_v2<op> <<<grid, block>>>(out, x);
#endif
  return out;
}


/* BROADCAST KERNELS */
//Broadcasts tensor and applies to another in element wise fasion.
//i.e. out [i, j] = op::op(A[i] + B[i, j])
template<typename op>
__global__ void
kernel_broadcast_apply(device_tensor<2> out,
		       const device_tensor<1> x, const device_tensor<2> y){
  size_t i = threadIdx.x;
  while(i<x.size[0]){
    for(size_t j=0; j<y.size[1]; j++){
      out.at(i, j) = op::op(x.at(i), y.at(i, j));
    }
    i += blockDim.x;
  }
}

template<typename op>
__global__ void
kernel_broadcast_apply_v2(device_tensor<2> out,
		       const device_tensor<1> x, const device_tensor<2> y){
  //size_t i = threadIdx.x;
  //while(i<x.size[0]){
  //  for(size_t j=0; j<y.size[1]; j++){
  //    out.at(i, j) = op::op(x.at(i), y.at(i, j));
  //  }
  //  i += blockDim.x;
  //}
  int rowidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowidx < x.size[0]) {
    for (int j = 0; j < y.size[1]; j++) {
       out.at_linear(rowidx * y.size[1] + j) = op::op(x.at(rowidx), y.at_linear(rowidx * y.size[1] + j));
    }
  }
}

/* BROADCAST KERNELS */
//Broadcasts tensor and applies to another in element wise fasion.
//i.e. out [i, j] = op::op(A[i, j] + B[i])
template<typename op>
__global__ void
kernel_broadcast_apply(device_tensor<2> out,
		       const device_tensor<2> x, const device_tensor<1> y){
  size_t i = threadIdx.x;
  while(i<x.size[0]){
    for(size_t j=0; j<x.size[1]; j++){
      out.at(i, j) = op::op(x.at(i, j), y.at(i));
    }
    i += blockDim.x;
  }
}

template<typename op>
__global__ void
kernel_broadcast_apply_v2(device_tensor<2> out,
		       const device_tensor<2> x, const device_tensor<1> y){
  //size_t i = threadIdx.x;
  //while(i<x.size[0]){
  //  for(size_t j=0; j<x.size[1]; j++){
  //    out.at(i, j) = op::op(x.at(i, j), y.at(i));
  //  }
  //  i += blockDim.x;
  //}
  int rowidx = threadIdx.x + blockDim.x * blockIdx.x;
  if (rowidx < x.size[0]) {
    for (int j=0; j < x.size[1]; j++) {
        out.at_linear(rowidx * x.size[1] + j) = op::op(x.at_linear(rowidx * x.size[1] + j), y.at(rowidx));
    }
  }
}

//GPU kernel wrapper for first broadcast kernel
template<typename op>
device_tensor< 2 > broadcast_apply(const device_tensor<2>& x, const device_tensor<1>& y){
  assert( x.size[0] == y.get_n_elems() );
  device_tensor<2> out(x.size);
  dim3 block(32);
  dim3 grid(x.size[0]/block.x);
#if 0
  kernel_broadcast_apply<op> <<<x.size[0]/256, 256>>>(out, x, y);
#else
  kernel_broadcast_apply_v2<op><<<grid, block>>>(out, x, y);
#endif
  return out;
}

//GPU kernel wrapper for second broadcast kernel
template<typename op>
device_tensor< 2 > broadcast_apply(const device_tensor<1>& x, const device_tensor<2>& y){
  assert( x.get_n_elems() == y.size[0] );
  device_tensor<2> out(y.size);
  dim3 block(32);
  dim3 grid(y.size[0]/block.x);
#if 0
  kernel_broadcast_apply<op> <<<y.size[0]/256, 256>>>(out, x, y);
#else
  kernel_broadcast_apply_v2<op><<<grid, block>>>(out, x, y);
#endif
  return out;
}
