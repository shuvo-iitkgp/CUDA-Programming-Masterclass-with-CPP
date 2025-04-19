#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda()
{
  printf("Hello CUDA WORLD \n");
}

int main()
{
  hello_cuda<<<20, 1>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}