#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>
#include "common.h"

__global__ void dynamic_parallelism_check(int size, int depth, int parent_id) {
    int tid = threadIdx.x; 
    int gid = blockIdx.x * blockDim.x + tid; 

    printf("Parent : %d - Depth : %d - tid : %d - gid : %d blockID: %d\n", parent_id ,depth, tid, gid, blockIdx.x); 

    if (size == 1){
      return;
    }

    if (threadIdx.x == 0){
      dynamic_parallelism_check<<<1, size / 2>>>(size/2, depth +1, blockIdx.x); 
    }
}



int main(int argc, char** argv) {

  dynamic_parallelism_check<<< 2, 8 >>> (8, 0, -1); 
  cudaDeviceSynchronize();
  cudaDeviceReset(); 
  
  return 0;
}
