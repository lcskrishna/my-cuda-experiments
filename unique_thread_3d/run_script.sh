mkdir -p bin
nvcc -o bin/unique_3d unique_3d.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/unique_3d
