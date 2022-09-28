mkdir -p bin
nvcc -o bin/unique_1d unique_1d.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/unique_1d
