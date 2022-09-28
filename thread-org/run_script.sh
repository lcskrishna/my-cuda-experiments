mkdir -p bin
nvcc -o bin/thread_org thread_org.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/thread_org
