#include "parallel.h"

float *d_src;
float *d_mean;
float *d_dest;
float *d_c_p;
float *d_u_p;

__global__ void thread_gwf(float *d_src, float *d_mean, float *d_dest, float *d_c_p, float *d_u_p,  double h, unsigned long p, int rows, int cols) {
    /**
     * Exit the threads that are outside the image boundary.
     */
    if (blockIdx.x == (int) cols/blockDim.x && threadIdx.x + blockIdx.x * blockDim.x >= cols) return;
    else if (blockIdx.y == (int) rows/blockDim.y && threadIdx.y + blockIdx.y * blockDim.y >= rows) return;
                
    __shared__ float c_p;
    __shared__ float u_p;

    unsigned long toffset = threadIdx.x + threadIdx.y * cols;
    unsigned long boffset = blockIdx.y * blockDim.y * cols + blockDim.x * blockIdx.x;

    int q = toffset + boffset;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        c_p = 0.f;
        u_p = 0.f;
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            *d_c_p = 0.f;
            *d_u_p = 0.f;
        }
    }

    __syncthreads();
    
    float mu_p = d_mean[p];

    double f_pq = float(exp(-1 * pow((mu_p - d_mean[q])/h, 2)));

    atomicAdd(&c_p, f_pq);
    atomicAdd(&u_p, d_src[q] * f_pq);

    __syncthreads();

    /**
     * Update the global memory per block.
     */
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(d_c_p, c_p);
        atomicAdd(d_u_p, u_p);
    }
    
    __syncthreads();

    if (q == 0) {
        float frac = *d_u_p/ *d_c_p;
        d_dest[p] =  frac;
    }
}

int NL_mean(float *h_src, float *h_mean, float *h_dest, double h, int rows, int cols, float *time) {
    cudaMalloc((void **) &d_src, sizeof(float) * rows * cols);
    cudaMalloc((void **) &d_mean, sizeof(float) * rows * cols);
    cudaMalloc((void **) &d_dest, sizeof(float) * rows * cols);
    cudaMalloc((void **) &d_c_p, sizeof(float));
    cudaMalloc((void **) &d_u_p, sizeof(float));

    cudaMemcpy(d_src, h_src, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, h_dest, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

    dim3 block(MAX_THREADS, MAX_THREADS, 1);
    dim3 grid(ceil(cols/MAX_THREADS), ceil(rows/MAX_THREADS));

    unsigned int p;
    unsigned int length = (rows-1) * (cols-1);
    unsigned int n = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            p = i * cols + j;
            thread_gwf<<<grid, block>>>(d_src, d_mean, d_dest, d_c_p, d_u_p, h, p, rows, cols);
        }

        if (float(p)/float(length) > n * .01f)
            printf("%d percent complete...\n", n++);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_dest, d_dest, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);

    cudaFree(d_u_p);    
    cudaFree(d_c_p);
    cudaFree(d_dest);
    cudaFree(d_mean);
    cudaFree(d_src);

    return 0;
}
