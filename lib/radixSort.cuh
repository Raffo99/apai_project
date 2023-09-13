#ifndef RADIX_SORT_H_
#define RADIX_SORT_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include "utils.cuh"

// CPU Functions
int getMax(int*, int);
void countingSort(int*, int, int);
void radix_sort_cpu(int*, int);

// GPU Functions
#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))


__global__ void prescan(int *, int *, int);
__global__ void add_block_sums(int* const, int* const, int* const, const size_t);
void sum_scan_blelloch(int* const, int* const, const size_t, int);
__global__ void radix_sort_local(int*, int*, int*, unsigned int, int*, unsigned int, unsigned int);
__global__ void global_shuffle(int*, int*, int*, int*, unsigned int, unsigned int, unsigned int);
void radix_sort(int* const, int* const, unsigned int, int);
#endif