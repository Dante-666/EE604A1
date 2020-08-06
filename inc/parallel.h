#ifndef __PARALLEL_H__
#define __PARALLEL_H__

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS 32.f


/**
 * Kernel to compute weights and sum for NL_mean in a multithreaded manner.
 */
__global__ void thread_gwf(float *d_src, float *d_mean, float *d_dest, float *d_c_p, float* d_u_p, double h, unsigned long p, int rows, int cols);

/**
 * Main caller for NL_mean, places the result in h_dest and running time in time.
 */
int NL_mean(float *h_src, float *h_mean, float *h_dest, double h, int rows, int cols, float *time);

#endif
