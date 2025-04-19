#include <cstdio>

__global__ void helloFromGPU()
{
    printf("Hello from the GPU thread!\n");
}

int main()
{
    // Launching the kernel with 1 block and 1 thread
    helloFromGPU<<<1, 1>>>();

    // Checking if the kernel launched properly
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Synchronize the device to make sure the output is flushed
    cudaDeviceSynchronize();

    return 0;
}
