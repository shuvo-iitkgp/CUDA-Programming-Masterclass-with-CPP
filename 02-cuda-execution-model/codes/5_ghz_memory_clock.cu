%%writefile occupancy.cu
#include <stdio.h>
#include <cuda_runtime.h>

// Your kernel
__global__ void occupancy_test(int *results) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int x1 = 1, x2 = 2, x3 = 3, x4 = 4, x5 = 5, x6 = 6, x7 = 7, x8 = 8;
    results[gid] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8;
}

int main() {
    const int blockSize = 128;  // Number of threads per block
    const int numBlocks = 80;   // Adjust based on your GPU
    const int numThreads = blockSize * numBlocks;

    // Allocate memory on device
    int *d_results;
    cudaMalloc((void**)&d_results, numThreads * sizeof(int));

    // Launch the kernel
    occupancy_test<<<numBlocks, blockSize>>>(d_results);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Occupancy calculation
    int maxActiveBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM,
        occupancy_test,
        blockSize,
        0  // dynamic shared memory
    );

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numSMs = prop.multiProcessorCount;
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

    float occupancy = (maxActiveBlocksPerSM * blockSize) / (float)maxThreadsPerSM;

    printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("Threads per SM: %d\n", maxActiveBlocksPerSM * blockSize);
    printf("Max threads per SM (hardware limit): %d\n", maxThreadsPerSM);
    printf("Occupancy: %.2f%%\n", occupancy * 100);

    // Free memory
    cudaFree(d_results);
    return 0;
}