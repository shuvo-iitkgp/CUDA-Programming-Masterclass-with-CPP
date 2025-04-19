#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void sum_array_gpu(int *a, int *b, int *c, int size)
{

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
  {
    c[gid] = a[gid] + b[gid];
  }
}
void sum_array_cpu(int *a, int *b, int *c, int size)
{
  for (int i = 0; i < size; i++)
  {
    c[i] = a[i] + b[i];
  }
}
void compare_arrays(int *a, int *b, int size)
{ // compare arrays
  for (int i = 0; i < size; i++)
  {
    if (a[i] != b[i])
    {
      printf("Arrays are different \n");
      return;
    }
  }
  printf("Arrays are the same \n");
}

int main()
{

  int size = 10000;
  int block_size = 128;

  int NO_BYTES = size * sizeof(int);

  int *h_a, *h_b, *gpu_results, *h_c;

  h_a = (int *)malloc(NO_BYTES);
  h_b = (int *)malloc(NO_BYTES);
  h_c = (int *)malloc(NO_BYTES);
  gpu_results = (int *)malloc(NO_BYTES);

  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++)
  {
    h_a[i] = (int)(rand() & 0xff);
  }
  for (int i = 0; i < size; i++)
  {
    h_b[i] = (int)(rand() & 0xff);
  }
  sum_array_cpu(h_a, h_b, h_c, size);

  // device pointer

  int *d_a, *d_b, *d_c;
  cudaMalloc((int **)&d_a, NO_BYTES);
  cudaMalloc((int **)&d_b, NO_BYTES);
  cudaMalloc((int **)&d_c, NO_BYTES);

  cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);

  dim3 block(block_size);
  dim3 grid((size / block.x) + 1);

  sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, size);
  cudaDeviceSynchronize();
  cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);

  compare_arrays(gpu_results, h_c, size);

  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);

  free(gpu_results);
  free(h_a);
  free(h_b);
  free(h_c);

  cudaDeviceReset();
  return 0;
}