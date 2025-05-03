#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "common.h"

__global__ void reduction_kernel(int *int_array, int *temp_array, int size) {
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int *i_data = int_array + blockDim.x * blockIdx.x ; //local data block pointer

    // Load data from global memory to shared memory
    s_data[tid] = (gid < size) ? int_array[gid] : 0;
    __syncthreads();

    // Tree-based reduction
    for (int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
        int index = 2 * offset * tid;
        if (index < blockDim.x) {
            s_data[index] += s_data[index + offset];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        temp_array[blockIdx.x] = s_data[0];
    }
}



int main() {
    printf("=== GPU Parallel Reduction ===\n");

    int size = 1024;
    int block_size = 128;
    int byte_size = size * sizeof(int);
    int grid_size = (size + block_size - 1) / block_size;

    // Host allocations
    int *h_input = (int*)malloc(byte_size);
    int *h_temp  = (int*)malloc(grid_size * sizeof(int));

    // Initialize
    initialize(h_input, size, INIT_ONE);
    int cpu_result = reduction_cpu(h_input, size);

    // Device allocations
    int *d_input, *d_temp;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMalloc((void**)&d_temp, grid_size * sizeof(int));
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Launch kernel with dynamic shared memory
    reduction_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_temp, size);
    cudaDeviceSynchronize();

    // Copy partial sums to host
    cudaMemcpy(h_temp, d_temp, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    int gpu_result = 0;
    for (int i = 0; i < grid_size; ++i) {
        gpu_result += h_temp[i];
    }

    compare_results(gpu_result, cpu_result);

    // Cleanup
    free(h_input);
    free(h_temp);
    cudaFree(d_input);
    cudaFree(d_temp);

    return 0;
}
