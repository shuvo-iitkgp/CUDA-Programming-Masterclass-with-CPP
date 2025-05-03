%%writefile reduction_kernel_complete_unrolling.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "common.h"

__global__ void reduction_kernel_complete_unrolling(int *int_array, int *temp_array, int size) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int *i_data = int_array + blockDim.x * blockIdx.x ; //local data block pointer


    if(blockDim.x == 1024 && tid < 512)
      i_data[tid] += i_data[tid + 512];
    __syncthreads();

    if(blockDim.x == 512 && tid < 256)
      i_data[tid] += i_data[tid +256];
    __syncthreads();

    if(blockDim.x == 256 && tid < 128)
      i_data[tid] += i_data[tid +128];
    __syncthreads();

    if(blockDim.x == 128 && tid < 64)
      i_data[tid] += i_data[tid +64];
    __syncthreads();

    if (tid < 32){
      volatile int *vsmem = i_data; // happen directly without any caches.
      vsmem[tid] += vsmem[tid + 32];
      vsmem[tid] += vsmem[tid + 16];
      vsmem[tid] += vsmem[tid + 8];
      vsmem[tid] += vsmem[tid + 4];
      vsmem[tid] += vsmem[tid +2];
      vsmem[tid] += vsmem[tid +1];
    }
    // Write result for this block to global memory
    if (tid == 0) {
        temp_array[blockIdx.x] = i_data[0];
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
    reduction_kernel_complete_unrolling<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_temp, size);
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