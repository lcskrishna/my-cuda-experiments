#include "common.h"

void initialize_array(int * array, int size, INIT_PARAM init_type)
{
    if (init_type == INIT_ONE_TO_TEN) {
        time_t t;
        srand((unsigned)time(&t));
        for (int i=0; i < size; i++) {
            array[i] = (int)(rand() & 0xFF) % 10;
        }
    }
    else if (init_type == INIT_ZERO) {
        for (int i=0; i < size; i++) {
            array[i] = 0;
        }
    }
    else if (init_type == INIT_RANDOM) {
        time_t t;
        srand((unsigned)time(&t));
        for (int i=0; i < size; i++) {
            array[i] = (int)(rand() & 0xFF);
        }
    }
    else if (init_type == INIT_ONE) {
        for (int i=0; i < size; i++) {
            array[i] = 1;
        }
    }
    else {
        std::cout << "ERROR: Unsupported parameter passed = " << init_type << std::endl;
    }
}

bool compare_arrays(int * a, int * b, int size)
{
    bool issame = true;
    for (int i=0; i < size; i++) {
        if (a[i] != b[i]) {
            issame = false;
            break;
        }
    }
    return issame;
}
