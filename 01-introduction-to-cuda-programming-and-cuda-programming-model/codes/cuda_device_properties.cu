#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

void query_device()
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0)
  {
    printf("No cuda support device found");
  }

  int devNo = 0;
  cudaDeviceProp iProp;
  cudaGetDeviceProperties(&iProp, devNo);

  printf("Device %d: %s \n", devNo, iProp.name);
  printf("number of multiprocessors: %d\n", iProp.multiProcessorCount);
  printf("Compute capabilitity: %d.%d\n", iProp.major, iProp.minor);
  printf("Total amount of global memory: %4.2f\n", iProp.totalGlobalMem / 1024.0);
  printf("Total amount of constant memory: %4.2f\n", iProp.totalConstMem / 1024.0);
  printf("Total amount of shared memory: %4.2f\n", iProp.sharedMemPerBlock / 1024.0);
}

int main()
{

  query_device();
  cudaDeviceReset();
  return 0;
}