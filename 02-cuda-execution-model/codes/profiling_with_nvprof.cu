%%writefile hello.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void my_kernel(int *d_array) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    d_array[idx] = idx * idx;
}

int main() {
    int N = 1024;
    int *d_array;

    cudaMalloc(&d_array, N * sizeof(int));

    // Launch kernel
    my_kernel<<<4, 256>>>(d_array);

    cudaDeviceSynchronize();
    cudaFree(d_array);

    std::cout << "Kernel executed!" << std::endl;
    return 0;
}