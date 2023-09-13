#include "../lib/bitonicSort.cuh"

// GPU Functions
__global__ void bitonic_sort_step(int * input, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i^j;

    if ((ixj) > i) {
        if ((i&k) == 0 && input[i] > input[ixj]) {
            int temp = input[i];
            input[i] = input[ixj];
            input[ixj] = temp;
        }

        if ((i&k) != 0 && input[i] < input[ixj]) {
            int temp = input[i];
            input[i] = input[ixj];
            input[ixj] = temp;
        }
    }
}

void bitonic_sort(int * input, int array_size, int num_threads) {
    int * input_cuda;

    unsigned int blocks_size = array_size / num_threads;

    if (checkCudaError(cudaMalloc(&input_cuda, sizeof(int) * array_size))) return;
    if (checkCudaError(cudaMemcpy(input_cuda, input, sizeof(int) * array_size, cudaMemcpyHostToDevice))) return;

    for (int k = 2; k <= array_size; k <<= 1) {
        for (int j = k>>1; j > 0; j = j>>1) {
            bitonic_sort_step<<<blocks_size, num_threads>>>(input_cuda, j, k);
        }
    }
    
    if (checkCudaError(cudaMemcpy(input, input_cuda, sizeof(int) * array_size, cudaMemcpyDeviceToHost))) return;
}

// CPU Functions
void exchange(int* num1, int* num2) {
    int temp;

    temp = *num1;
    *num1 = *num2;
    *num2 = temp;
}

void compare(int* arr, int i, int j, int dir) {
    if (dir == (arr[i] > arr[j])) {
        exchange(&arr[i], &arr[j]);
    }
}

void bitonic_merge(int* array, int low, int c, int dir) {
    int k = 0;
    int i = 0;

    if (c > 1) {
        k = c / 2;
        i = low;
        while (i < low + k) {
            compare(array, i, i + k, dir);
            i++;
        }

        bitonic_merge(array, low, k, dir);
        bitonic_merge(array, low + k, k, dir);
    }
}

void recbitonic(int* array, int low, int v, int dir) {
    int k = 0;

    if (v > 1) {
        k = v / 2;
        recbitonic(array, low, k, 1);
        recbitonic(array, low + k, k, 0);
        bitonic_merge(array, low, v, dir);
    }
}