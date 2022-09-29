mkdir -p bin
nvcc -o bin/memory_transfer memory_transfer.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/memory_transfer
