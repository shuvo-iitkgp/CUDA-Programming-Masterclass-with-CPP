#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_details()
{
  printf("threadIdx.x: %d, threadIdx.y : %d, threadIdx.z : %d, blockIdx.x: %d, blockIdx.y : %d, blockIdx.z : %d, blockDim.x: %d, blockDim.y: %d,blockDim.z: %d, gridDim.x: %d, gridDim.y: %d, gridDim.z: %d \n",
         threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
  int nx, ny, nz;
  nx = 4;
  ny = 4;
  nz = 4;
  dim3 block(2, 2, 2);
  dim3 grid(nx / block.x, ny / block.y, nz / block.y);
  print_details<<<grid, block>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}