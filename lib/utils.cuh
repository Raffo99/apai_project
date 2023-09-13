#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

int checkCudaError(cudaError_t);
int checkIncremental(int*, int);
int checkSorted(int*, int);
#endif