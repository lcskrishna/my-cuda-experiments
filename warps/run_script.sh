mkdir -p bin
nvcc -o bin/warp_id_thread_id warp_id_thread_id.cu -I /usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/
./bin/warp_id_thread_id
