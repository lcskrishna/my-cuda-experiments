mkdir -p bin
nvcc -o bin/print_3d_threads print_3d_threads.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/print_3d_threads
