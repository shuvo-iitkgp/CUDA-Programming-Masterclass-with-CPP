#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void mem_trs_test(int *input, int size)
{

  int tid = threadIdx.z * (blockDim.x * blockDim.y) +
            threadIdx.y * blockDim.x +
            threadIdx.x;

  int block_offset = blockIdx.x +
                     blockIdx.y * gridDim.x +
                     blockIdx.z * (gridDim.x * gridDim.y);
  int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  int gid = threads_per_block * block_offset + tid;

  if (gid < size)
  {
    printf("tid : %d , grid : %d , value : %d \n", tid, gid, input[gid]);
  }
}

int main()
{

  int size = 64;
  int byte_size = size * sizeof(int);
  int *h_input;
  h_input = (int *)malloc(byte_size);

  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++)
  {
    h_input[i] = (int)(rand() & 0xff);
  }

  int *d_input;
  cudaMalloc((void **)&d_input, byte_size);

  cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
  int nx, ny, nz;
  nx = 4;
  ny = 4;
  nz = 4;

  dim3 block(2, 2, 2);
  dim3 grid(nx / block.x,
            ny / block.y,
            nz / block.z);

  mem_trs_test<<<grid, block>>>(d_input, size);

  cudaDeviceSynchronize();
  cudaFree(d_input);
  free(h_input);

  cudaDeviceReset();
  return 0;
}