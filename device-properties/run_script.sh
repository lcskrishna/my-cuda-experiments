mkdir -p bin
nvcc -o bin/device_query device_query.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/device_query
