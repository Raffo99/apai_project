#ifndef BITONIC_SORT_H_
#define BITONIC_SORT_H_

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>
#include "utils.cuh"

// GPU Functions
__global__ void bitonic_sort_step(int *, int, int);
void bitonic_sort(int*, int, int);

// CPU Functions
void exchange(int*, int*);
void compare(int*, int, int, int);
void bitonic_merge(int*, int, int, int);
void recbitonic(int*, int, int, int);
#endif