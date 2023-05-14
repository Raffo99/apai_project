#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <malloc.h>

#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define MAX_BLOCK_SIZE 256

int checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("%s", cudaGetErrorString(error));
        return true;
    }

    return false;
}

__global__ void prescan(int *g_odata, int *g_idata, int n) { 
    extern __shared__ int temp[];
    int thread_id = threadIdx.x;
    int offset = 1;

    int ai = thread_id; 
    int bi = thread_id + (n/2); 
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai); 
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai]; 
    temp[bi + bankOffsetB] = g_idata[bi]; 

    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (thread_id < d) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    } 

    if (thread_id == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thread_id < d) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            int t = temp[ai]; 
            temp[ai] = temp[bi]; 
            temp[bi] += t;
        }
    }
    __syncthreads();

    g_odata[ai] = temp[ai + bankOffsetA]; 
    g_odata[bi] = temp[bi + bankOffsetB]; 
}

__global__ void add_block_sums(int* const out, int* const in, int* const block_sums, const size_t number_elements) {
    int block_sums_value = block_sums[blockIdx.x];
    unsigned int work_id = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (work_id < number_elements) {
        out[work_id] = in[work_id] + block_sums_value;
        if (work_id + blockDim.x < number_elements) out[work_id + blockDim.x] = in[work_id + blockDim.x] + block_sums_value;
    }
}

void sum_scan_blelloch(int* const out, int* const in, const size_t number_elements) {
    if (checkCudaError(cudaMemset(out, 0, sizeof(int) * number_elements))) return;

    unsigned int block_size = MAX_BLOCK_SIZE / 2;
    unsigned int max_elements_per_block = block_size * 2;
    unsigned int grid_size = number_elements / max_elements_per_block;
    
    if (number_elements % max_elements_per_block != 0) grid_size += 1;

    unsigned int shared_size = max_elements_per_block + ((max_elements_per_block - 1) >> LOG_NUM_BANKS);

    int* block_sums;
    if (checkCudaError(cudaMalloc(&block_sums, sizeof(int) * grid_size))) return;
    if (checkCudaError(cudaMemset(block_sums, 0, sizeof(int) * grid_size))) return;

    prescan<<<grid_size, block_size, sizeof(int) * shared_size>>>(out, in, max_elements_per_block);

    if (grid_size <= max_elements_per_block) {
        int* dummy_blocks_sums;
        if (checkCudaError(cudaMalloc(&dummy_blocks_sums, sizeof(int)))) return;
        if (checkCudaError(cudaMemset(dummy_blocks_sums, 0, sizeof(int)))) return;

        prescan<<<1, block_size, sizeof(int) * shared_size>>>(block_sums, block_sums, max_elements_per_block);
        if (checkCudaError(cudaFree(dummy_blocks_sums))) return;
    } else {
        int* input_block_sums;
        if (checkCudaError(cudaMalloc(&input_block_sums, sizeof(int) * grid_size))) return;
        if (checkCudaError(cudaMemcpy(input_block_sums, block_sums, sizeof(int) * grid_size, cudaMemcpyDeviceToDevice))) return;

        sum_scan_blelloch(block_sums, input_block_sums, grid_size);
        if (checkCudaError(cudaFree(input_block_sums))) return;
    }

    add_block_sums<<<grid_size, block_size>>>(out, out, block_sums, number_elements);
}

__global__ void radix_sort_local(int* out, int* prefix_sums, int* block_sums, unsigned int bit, int* in, unsigned int in_len, unsigned int max_elements_per_block) {
    extern __shared__ int shared_memory[];
    int* shared_data = shared_memory;
    unsigned int mask_out_len = max_elements_per_block + 1;
    int* mask_out = &shared_data[max_elements_per_block];
    int* merged_scan_mask_out = &mask_out[mask_out_len];
    int* mask_out_sums = &merged_scan_mask_out[max_elements_per_block];
    int* scan_mask_out_sums = &mask_out_sums[4];

    unsigned int thread_id = threadIdx.x;

    unsigned int work_id = max_elements_per_block * blockIdx.x + thread_id;
    shared_data[thread_id] = (work_id < in_len) ? in[work_id] : 0;
    
    __syncthreads();

    int thread_data = shared_data[thread_id];
    int extract_bits = (thread_data >> bit) & 3;

    for (unsigned int b = 0; b < 4; b++) {
        mask_out[thread_id] = 0;
        if (thread_id == 0)
            mask_out[mask_out_len - 1] = 0;
        
        __syncthreads();

        bool bits_equals_b = false;
        if (work_id < in_len) {
            bits_equals_b = b == extract_bits;
            mask_out[thread_id] = bits_equals_b;
        }
        __syncthreads();

        int partner = 0;
        int sum = 0;
        for (unsigned int d = 0; d < (unsigned int) log2f(max_elements_per_block); d++) {
            partner = thread_id - (1 << d);
            sum = (partner >= 0) ? mask_out[thread_id] + mask_out[partner] : mask_out[thread_id];
            __syncthreads();
            mask_out[thread_id] = sum;
            __syncthreads();
        }

        int work_val = 0;
        work_val = mask_out[thread_id];
        __syncthreads();
        mask_out[thread_id + 1] = work_val;
        __syncthreads();

        if (thread_id == 0) {
            mask_out[0] = 0;
            int total_sum = mask_out[mask_out_len - 1];
            mask_out_sums[b] = total_sum;
            block_sums[b * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();

        if (bits_equals_b && (work_id < in_len))
            merged_scan_mask_out[thread_id] = mask_out[thread_id];
        __syncthreads();
    }

    if (thread_id == 0) {
        int run_sum = 0;
        for (unsigned int i = 0; i < 4; i++) {
            scan_mask_out_sums[i] = run_sum;
            run_sum += mask_out_sums[i];
        }
    }

    __syncthreads();

    if (work_id < in_len) {
        int prefix_sum = merged_scan_mask_out[thread_id];
        int new_pos = prefix_sum + scan_mask_out_sums[extract_bits];
        __syncthreads();

        shared_data[new_pos] = thread_data;
        merged_scan_mask_out[new_pos] = prefix_sum;
        __syncthreads();

        prefix_sums[work_id] = merged_scan_mask_out[thread_id];
        out[work_id] = shared_data[thread_id];
    }
}

__global__ void global_shuffle(int* out, int* in, int* scan_block_sums, int* prefix_sums, unsigned int bit, unsigned int in_len, unsigned int max_elements_per_block) {
    unsigned int thread_id = threadIdx.x;
    unsigned int work_id = max_elements_per_block * blockIdx.x + thread_id;

    if (work_id < in_len) {
        int data = in[work_id];
        int extract_bits = (data >> bit) & 3;
        int prefix_sum = prefix_sums[work_id];
        int global_position = scan_block_sums[extract_bits * gridDim.x + blockIdx.x] + prefix_sum;
        __syncthreads();
        out[global_position] = data;
    }
}

void radix_sort(int* const out, int* const in, unsigned int in_len) {
    unsigned int block_size = MAX_BLOCK_SIZE;
    unsigned int max_elements_per_block = block_size;
    unsigned int grid_size = in_len / max_elements_per_block;

    if (in_len % max_elements_per_block != 0)
        grid_size += 1;
    
    int* prefix_sums;
    unsigned int prefix_sums_len = in_len;

    if (checkCudaError(cudaMalloc(&prefix_sums, sizeof(int) * prefix_sums_len))) return;
    if (checkCudaError(cudaMemset(prefix_sums, 0, sizeof(int) * prefix_sums_len))) return;

    int* block_sums;
    unsigned int block_sums_len = 4 * grid_size;

    if (checkCudaError(cudaMalloc(&block_sums, sizeof(int) * block_sums_len))) return;
    if (checkCudaError(cudaMemset(block_sums, 0, sizeof(int) * block_sums_len))) return;

    int* scan_block_sums;
    if (checkCudaError(cudaMalloc(&scan_block_sums, sizeof(int) * block_sums_len))) return;
    if (checkCudaError(cudaMemset(scan_block_sums, 0, sizeof(int) * block_sums_len))) return;

    // unsigned int data_len = max_elements_per_block;
    unsigned int mask_out_len = max_elements_per_block + 1;
    unsigned int merged_scan_mask_out_len = max_elements_per_block;
    unsigned int mask_out_sums_len = max_elements_per_block;
    unsigned int scan_mask_out_sums_len = 4;
    unsigned int shared_size = (mask_out_len + merged_scan_mask_out_len + mask_out_sums_len + scan_mask_out_sums_len) * sizeof(int);

    for (unsigned int bit = 0; bit <= 30; bit += 2) {
        radix_sort_local<<<grid_size, block_size, shared_size>>>(out, prefix_sums, block_sums, bit, in, in_len, max_elements_per_block);
        sum_scan_blelloch(scan_block_sums, block_sums, block_sums_len);
        global_shuffle<<<grid_size, block_size>>>(in, out, scan_block_sums, prefix_sums, bit, in_len, max_elements_per_block);
    }

    if (checkCudaError(cudaMemcpy(out, in, sizeof(unsigned int) * in_len, cudaMemcpyDeviceToDevice))) return;
    if (checkCudaError(cudaFree(scan_block_sums))) return;
    if (checkCudaError(cudaFree(block_sums))) return;
    if (checkCudaError(cudaFree(prefix_sums))) return;
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

int main(int argc, char **argv) {
    int array_size = atoi(argv[1]);
    int verbose = atoi(argv[2]);

	srand(time(NULL));
    printf("%d\n", array_size);

    int *to_sort_array;
    int *sorted_array;
    int *input_cuda;
    int *output_cuda;

    // Generate the array to sort
    to_sort_array = (int *) malloc(sizeof(int) * array_size);
    sorted_array = (int *) malloc(sizeof(int) * array_size);
    for (int i = 0; i < array_size; i++)
        to_sort_array[i] = (rand() % 300) + 1; //(array_size - 1) - i; 
    
    // Print the starting array
    if (verbose) {
        printf("Starting array: ");
        for (int i = 0; i < array_size; i++)
            printf("%d ", to_sort_array[i]);
        printf("\n");
    }

    if (checkCudaError(cudaMalloc(&input_cuda, sizeof(unsigned int) * array_size))) return -1;
    if (checkCudaError(cudaMemcpy(input_cuda, to_sort_array, sizeof(unsigned int) * array_size, cudaMemcpyHostToDevice))) return -1;
    if (checkCudaError(cudaMalloc(&output_cuda, sizeof(unsigned int) * array_size))) return -1;

    cudaEvent_t beginEvent;
    cudaEvent_t endEvent;

    cudaEventCreate(&beginEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(beginEvent);

    radix_sort(output_cuda, input_cuda, array_size);

    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);

    float timeValue = 0;
    cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
    
    cudaEventDestroy(beginEvent);
    cudaEventDestroy(endEvent);
    
    if (checkCudaError(cudaMemcpy(sorted_array, output_cuda, sizeof(int) * array_size, cudaMemcpyDeviceToHost))) return -1;

    // Print the sorted array
    if (verbose) {
        printf("\nRadix sort results: ");
        for (int i = 0; i < array_size; i++)
            printf("%d ", sorted_array[i]);
        printf("\n\n");
    }

    printf("Sorted: %d\n", checkSorted(sorted_array, array_size));
    printf("Incremental: %d\n", checkIncremental(sorted_array, array_size));
    printf("GPU Time: %.2f\n", timeValue);
    delete[] to_sort_array;
    delete[] sorted_array;
    cudaFree(input_cuda);
    cudaFree(output_cuda);
    cudaDeviceReset();
    
    return EXIT_SUCCESS;
}