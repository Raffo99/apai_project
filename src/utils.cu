#include "../lib/utils.cuh"

int checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("%s", cudaGetErrorString(error));
        return true;
    }

    return false;
}

int checkIncremental(int *array, int array_size) {
    for (int i = 0; i < array_size - 1; i++)
        if (array[i] + 1 != array[i + 1]) 
            return false;
    
    return true;    
}

int checkSorted(int *array, int array_size) {
    for (int i = 0; i < array_size - 1; i++)
        if (array[i] > array[i + 1])
            return 0;
    return 1;
}