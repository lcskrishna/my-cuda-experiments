mkdir -p bin
nvcc -o bin/sum_three_arrays sum_three_arrays.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/sum_three_arrays
