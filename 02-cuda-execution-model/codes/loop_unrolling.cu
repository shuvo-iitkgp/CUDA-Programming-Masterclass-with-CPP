%%writefile parallel_reduction_wrap_unrolling.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "common.h"

__global__ void reduction_unrolling_blocks2(int *int_array, int *temp_array, int size) {
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
    int index = BLOCK_OFFSET + tid;

    // Load two elements per thread from global memory
    int sum = 0;
    if (index < size) {
        sum = int_array[index];
        if (index + blockDim.x < size) {
            sum += int_array[index + blockDim.x];
        }
    }

    s_data[tid] = sum;
    __syncthreads();

    // Standard reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        temp_array[blockIdx.x] = s_data[0];
    }
}


__global__ void reduction_unrolling_blocks4(int *int_array, int *temp_array, int size) {
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
    int index = BLOCK_OFFSET + tid;

    // Load data from global memory to shared memory
    s_data[tid] = (gid < size) ? int_array[gid] : 0;
    __syncthreads();

    if ((index + 3 * blockDim.x) < size){
      s_data[index] += s_data[index + blockDim.x];
      int a1 = s_data[index];
      int a2 = s_data[index + blockDim.x];
      int a3 = s_data[index + 2 * blockDim.x];
      int a4 = s_data[index + 3 * blockDim.x];

      s_data[index] = a1+ a2 + a3+ a4;
    }
    __syncthreads();
    // Tree-based reduction
    for (int offset = blockDim.x / 2; offset > 0 ; offset = offset / 2) {

        if (tid < offset) {
            s_data[tid] += s_data[tid + offset];
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
    // int grid_size = (size + block_size - 1) / block_size; // for one block kernel
    int grid_size = (size + block_size * 2 - 1) / (block_size * 2); // for 2 block kernel

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
    reduction_unrolling_blocks2<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_temp, size);
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