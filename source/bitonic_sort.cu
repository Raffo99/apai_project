#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define THREADS 1024 // 2^9
#define BLOCKS 1024// 2^15
#define NUM_VALS THREADS*BLOCKS

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

void bitonic_sort(int * input, int array_size) {
    int * input_cuda;

    // unsigned int blocks_size = array_size / THREADS_PER_BLOCKS;

    if (checkCudaError(cudaMalloc(&input_cuda, sizeof(int) * array_size))) return;
    if (checkCudaError(cudaMemcpy(input_cuda, input, sizeof(int) * array_size, cudaMemcpyHostToDevice))) return;

    for (int k = 2; k <= array_size; k <<= 1) {
        for (int j = k>>1; j > 0; j = j>>1) {
            bitonic_sort_step<<<BLOCKS, THREADS>>>(input_cuda, j, k);
        }
    }
    
    if (checkCudaError(cudaMemcpy(input, input_cuda, sizeof(int) * array_size, cudaMemcpyDeviceToHost))) return;
}

int main(int argc, char **argv) {
    int array_size = NUM_VALS;
    int verbose = atoi(argv[2]);

	srand(time(NULL));

    int * to_sort_array;

    // Generate the array to sort
    to_sort_array =  (int *) malloc(sizeof(int) * array_size);
    for (int i = 0; i < array_size; i++)
        to_sort_array[i] = (rand() % 300) + 1; // (array_size - 1) - i; 

    // Print the starting array
    if (verbose) {
        printf("Starting array: ");
        for (int i = 0; i < array_size; i++)
            printf("%d ", to_sort_array[i]);
        printf("\n");
    }

    cudaEvent_t beginEvent;
    cudaEvent_t endEvent;

    cudaEventCreate(&beginEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(beginEvent);

    bitonic_sort(to_sort_array, array_size);

    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);

    float timeValue = 0;
    cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
    
    cudaEventDestroy(beginEvent);
    cudaEventDestroy(endEvent);

    // Print the sorted array
    if (verbose) {
        printf("Bitonic sort result: ");
        for (int i = 0; i < array_size; i++)
            printf("%d ", to_sort_array[i]);
        printf("\n\n");
    }

    printf("Sorted: %d\n", checkSorted(to_sort_array, array_size));
    printf("Incremental: %d\n", checkIncremental(to_sort_array, array_size));
    printf("GPU Time: %.2f\n", timeValue);

    delete[] to_sort_array;
    cudaDeviceReset();

    return EXIT_SUCCESS;
}