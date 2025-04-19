
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void unique_gid_2Dcalculation(int *data)
{
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * blockDim.x;
  int row_offset = blockDim.x * gridDim.x * blockIdx.y;
  int gid = row_offset + block_offset + tid;
  printf("blockIdx.x: %d,blockIdx.y: %d, threadIdx.x : %d, gid : %d, -  value : %d \n ",
         blockIdx.x, blockIdx.y, threadIdx.x, gid, data[gid]);
}

int main()
{

  int array_size = 16;
  int array_byte_size = array_size * sizeof(int);
  int h_data[] = {24, 4, 23, 43, 1, 8, 4, 42, 1, 4, 2, 3, 32, 43, 3, 89};
  for (int i = 0; i < array_size; i++)
  {
    printf("%d ", h_data[i]);
  }
  printf("\n\n");
  int *d_data;
  cudaMalloc((void **)&d_data, array_byte_size);
  cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

  dim3 block(4);
  dim3 grid(2, 2);

  unique_gid_2Dcalculation<<<grid, block>>>(d_data);

  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}