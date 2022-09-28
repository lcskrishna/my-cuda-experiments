mkdir -p bin
nvcc -o bin/unique_2d unique_2d.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/unique_2d
