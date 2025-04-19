#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

#define gpuErrchck(ans)                   \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  };

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d \n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

__global__ void sum_array_gpu(int *a, int *b, int *c, int *d, int size)
{

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
  {
    d[gid] = a[gid] + b[gid] + c[gid];
  }
}
void sum_array_cpu(int *a, int *b, int *c, int *d, int size)
{
  for (int i = 0; i < size; i++)
  {
    d[i] = a[i] + b[i] + c[i];
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

  int size = 4194304;
  int block_size = 512;

  int NO_BYTES = size * sizeof(int);

  int *h_a, *h_b, *h_c, *gpu_result, *cpu_result;

  h_a = (int *)malloc(NO_BYTES);
  h_b = (int *)malloc(NO_BYTES);
  h_c = (int *)malloc(NO_BYTES);

  gpu_result = (int *)malloc(NO_BYTES);
  cpu_result = (int *)malloc(NO_BYTES);

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
  for (int i = 0; i < size; i++)
  {
    h_c[i] = (int)(rand() & 0xff);
  }

  memset(gpu_result, 0, NO_BYTES);
  memset(cpu_result, 0, NO_BYTES);

  clock_t cpu_start, cpu_end;
  cpu_start = clock();

  sum_array_cpu(h_a, h_b, h_c, cpu_result, size);
  cpu_end = clock();

  // device pointer
  int *d_a, *d_b, *d_c, *d_d;
  gpuErrchck(cudaMalloc((int **)&d_a, NO_BYTES));
  gpuErrchck(cudaMalloc((int **)&d_b, NO_BYTES));
  gpuErrchck(cudaMalloc((int **)&d_c, NO_BYTES));
  gpuErrchck(cudaMalloc((int **)&d_d, NO_BYTES));

  dim3 block(block_size);
  dim3 grid((size / block.x) + 1);

  clock_t htod_start, htod_end;
  htod_start = clock();

  cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, NO_BYTES, cudaMemcpyHostToDevice);

  htod_end = clock();
  clock_t gpu_start, gpu_end;
  gpu_start = clock();
  sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, d_d, size);
  cudaDeviceSynchronize();
  gpu_end = clock();
  clock_t dtoh_start, dtoh_end;
  dtoh_start = clock();
  cudaMemcpy(gpu_result, d_d, NO_BYTES, cudaMemcpyDeviceToHost);
  dtoh_end = clock();

  compare_arrays(cpu_result, gpu_result, size);

  printf("SUM ARRAY CPU execution time : %4.6f \n", (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

  printf("SUM ARRAY GPU execution time : %4.6f \n",
         (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));

  printf("htod mem transfer time : %4.6f \n",
         (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));

  printf("dtod mem transfer time : %4.6f \n",
         (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));

  printf("SUm array gpu total execution time : %4.6f \n",
         (double)((double)(dtoh_end - htod_start) / CLOCKS_PER_SEC));

  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);

  free(gpu_result);
  free(h_a);
  free(h_b);
  free(cpu_result);

  cudaDeviceReset();
  return 0;
}