%%writefile reduction_dynamic_programming.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "common.h"

__global__ void gpuRecursiveReduce(int *input, int *output, int size) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load global data into shared memory
    sdata[tid] = (gid < size) ? input[gid] : 0;
    __syncthreads();

    // Interleaved reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Store per-block result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }

    // Recursively reduce further
    if (tid == 0 && blockIdx.x == 0) {
        int num_blocks = (size + blockDim.x - 1) / blockDim.x;
        if (num_blocks > 1) {
            int new_blocks = (num_blocks + blockDim.x - 1) / blockDim.x;
            gpuRecursiveReduce<<<new_blocks, blockDim.x, blockDim.x * sizeof(int)>>>(output, output, num_blocks);
        }
    }
}

int main() {
    const int size = 1024;
    const int block_size = 128;
    const int bytes = size * sizeof(int);

    // Allocate and initialize host memory
    int *h_input = (int *)malloc(bytes);
    for (int i = 0; i < size; i++) h_input[i] = 1;
    int cpu_result = reduction_cpu(h_input, size);  // should return 1024

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);  // reuse same buffer for recursive output

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Allow recursive launches
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 8);

    int grid_size = (size + block_size - 1) / block_size;

    // Launch recursive reduction kernel
    gpuRecursiveReduce<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    // Read final result from d_output[0]
    int gpu_result = 0;
    cudaMemcpy(&gpu_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    //printf("GPU RESULT : %d, CPU RESULT : %d\n", gpu_result, cpu_result);
    compare_results(gpu_result, cpu_result);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
