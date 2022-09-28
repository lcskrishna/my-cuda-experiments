nvcc -o bin/hello_world hello_cuda.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/hello_world
