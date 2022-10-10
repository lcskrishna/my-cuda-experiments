//Simple math operations to be used with device_patterns.cuh

struct square_op{
  __host__ __device__ static inline float op(const float& a){
    return a*a;
  }
};

struct sinh_op{
  __host__ __device__ static inline float op(const float& a){
#if 0
    return sinh( (double) a );
#else
    return sinhf(a);
#endif
  }
};

struct square_root_op{
  __host__ __device__ static inline float op(const float& a){
    return sqrtf(a);
  }
};

struct add_op{
  __host__ __device__ static inline float op(const float& a, const float& b){
    return a+b;
  }

  //Init value for reduction use of this op
  __host__ __device__ static inline float init(){
    return 0.0;
  }
};

struct mul_op{
  __host__ __device__ static inline float op(const float& a, const float& b){
    return a*b;
  }
};

struct div_op{
  __host__ __device__ static inline float op(const float& a, const float& b){
#if 1
    return a/b;
#else
    return __fdiv_rz(a, b);
#endif
  }
};

struct sub_op{
  __host__ __device__ static inline float op(const float& a, const float& b){
    return a-b;
  }
};
