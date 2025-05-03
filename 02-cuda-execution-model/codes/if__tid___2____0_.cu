%%writefile hello.cu
#include <cstdio>

__global__ void code_without_divergence() {

  int gid = blockIdx.x * blockDim.x + threadIdx.x ;
  float a, b;
  a=b=0;
  int warp_id = gid / 32;
  if (warp_id%2 ==0){
    a = 100.0;
    b= 50.0;
  }
  else{
    a = 200;
    b = 75;
  }
  // Add dummy computation
  for (int i = 0; i < 1000; ++i) {
        a = a * 1.00001f + b * 0.99999f;
    }
}

__global__ void divergence_code() { // consecutive threads will take difference part of executions

  int gid = blockIdx.x * blockDim.x + threadIdx.x ;
  float a, b;
  a=b=0;
  if (gid%2 ==0){
    a = 100.0;
    b= 50.0;
  }
  else{
    a = 200;
    b = 75;
  }
  // Add dummy computation
  for (int i = 0; i < 1000; ++i) {
        a = a * 1.00001f + b * 0.99999f;
    }
}

int main() {
  printf("\n-----------------------WARP DIVERGENCE EXAMPLE-------------------------------\n\n");
  int size = 1<<22;
  dim3 block_size(128);
  dim3 grid_size((size + block_size.x - 1) / block_size.x);

  code_without_divergence<<<grid_size,block_size>>>();
  cudaDeviceSynchronize();


  divergence_code<<<grid_size,block_size>>>();
  // Synchronize the device to make sure the output is flushed
  cudaDeviceSynchronize();

    return 0;
}