#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../lib/radixSort.cuh"
#include "../lib/bitonicSort.cuh"
#include "../lib/utils.cuh"

int main(int argc, char *argv[]) {
    int algorithm = atoi(argv[1]) - 1;
    int num_threads = atoi(argv[2]);
    int array_size = atoi(argv[3]);
    int verbose = atoi(argv[4]);

    char algorithms[][150] = {"Parallel Bitonic Sort",
    "Sequential Bitonic Sort", "Parallel Radix Sort", "Sequential Radix Sort"};

	srand(time(NULL));

    int *to_sort_array;
    int *input_cuda;
    int *output_cuda;

    // Generate the array to sort
    to_sort_array =  (int *) malloc(sizeof(int) * array_size);
    for (int i = 0; i < array_size; i++)
        to_sort_array[i] = (array_size - 1) - i; 

    // Print the starting array
    if (verbose) {
        printf("Starting array: ");
        for (int i = 0; i < array_size; i++)
            printf("%d ", to_sort_array[i]);
        printf("\n");
    }

    cudaEvent_t beginEvent;
    cudaEvent_t endEvent;

    struct timespec start, end;

    cudaEventCreate(&beginEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(beginEvent);

    if (algorithm == 0) {
        bitonic_sort(to_sort_array, array_size, num_threads);
    } else if (algorithm == 1) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        recbitonic(to_sort_array, 0, array_size, 1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    } else if (algorithm == 2) {
        if (checkCudaError(cudaMalloc(&input_cuda, sizeof(unsigned int) * array_size))) return -1;
        if (checkCudaError(cudaMemcpy(input_cuda, to_sort_array, sizeof(unsigned int) * array_size, cudaMemcpyHostToDevice))) return -1;
        if (checkCudaError(cudaMalloc(&output_cuda, sizeof(unsigned int) * array_size))) return -1;
        radix_sort(output_cuda, input_cuda, array_size, num_threads);
        if (checkCudaError(cudaMemcpy(to_sort_array, output_cuda, sizeof(int) * array_size, cudaMemcpyDeviceToHost))) return -1;
    } else {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        radix_sort_cpu(to_sort_array, array_size);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    }
    
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);

    float timeValue = 0;
    if (algorithm % 2 != 0) timeValue = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    else cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
    
    cudaEventDestroy(beginEvent);
    cudaEventDestroy(endEvent);

    // Print the sorted array
    if (verbose) {
        printf("%s result: ", algorithms[algorithm]);
        for (int i = 0; i < array_size; i++)
            printf("%d ", to_sort_array[i]);
        printf("\n\n");
    }

    printf("%s (%d)\n", algorithms[algorithm], array_size);
    printf("Sorted: %d\n", checkSorted(to_sort_array, array_size));
    printf("Incremental: %d\n", checkIncremental(to_sort_array, array_size));
    printf("Time: %.2f\n\n", timeValue);

    delete[] to_sort_array;
    cudaDeviceReset();

    return EXIT_SUCCESS;
}