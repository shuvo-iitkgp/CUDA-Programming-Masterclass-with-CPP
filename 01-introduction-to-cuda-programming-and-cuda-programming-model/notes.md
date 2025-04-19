## HOW TO USE THE CUDA AND COMPILE IT AND GET OUTPUT

## CUDA JOURNEY BEGINSSS

Basic steps of A CUDA PROGRAM

1. init of data from cpu
2. transfer data from cpu to gpu
3. kernel launch with needed grid/block size
4. transfer results back to cpu context from gpu context
5. reclaim the memory from both cpu and gpu

GRID: collection of all threads launch for a kernel

BLOCK : threads in a gird is organized in to groups of blocks

## Limitation of block size

y<= 1024 and x <= 1024 and z <= 64

x * y * z <= 1024

## Organization of threads in a CUDA Program 1

### program 2

**blockIdx** : CUDA runtime uniquely intialized blockIdx variable for each thread depending on the coordinates of the belonging thread block in the grid

blockidx is dim3 type variable


**blockDim** : variable consist number of threads in each dimension of a thread block . Notice all the thread block in a grid have same block size , so this variable value is same for all the threads in a grid.

blockDim is dim3 type variable

**GridDim** variable consist number of thread blocks in each dimension of a grid.

GridDim is dim3 type variable.

## Assignment 1

Print value of threadIdx, blockIDx, gridDim variables for 3D grid which has 4 threads in all X, Y, and Z dimension and thread block size will be 2 threads in each dimension as well.

## Unique index calculation for thread in a grid

Array

23 9 4 54 64 12 1 33

0 1 2 3 4 5 6 7 <-- ThreadIdx.x
Grid

A B C D E F G H

Now we launch 2 thread blocks in one grid


ThreadIdx.x 0 1 2 3 0 1 2 3

ThreadBlock [A B C D] [E F G H]

We add an offset of 4 to get the id of the number

gid = tid + offest

gid = tid + blockIdx.x * blockDim.x


## Unique Calculation of gid in 2D array


Index= row offset + block offset + tid

number of threads in one thread block row  = gridDim.x * blockDim.x

gid = gridDim.x * blockDim.x * blockIdx.y
+ blockIdx.X * blockDim.x + threadIdx

Row offset = gridDim.x * blockDim.x * blockIdx.y

Block Offset = blockIdx.X * blockDim.X



- the memory access pattern is going to depend on the way we calculate our global index
- we usually prefer to calculate global indices in a way that, threads with in same thread block access consecutive memory locations or consecutive elements in an array

if the block has more than 1 dimesnions:

[[0, 1],

[2, 3]]


tid = threadIdx.y * blockDim.x + threadIdx.x

block_offset = number of thread in a block * blockIdx.x

row_offset = number fo threads in a row * blockIdx.y


## CUDA memory transfer

Initialize Data -> host logic -> Wating for the GPU results

Memory transfer occurs in both steps

We can transfer memory from host to device by  🇰

```cudaMemCpy(destination ptr, source ptr, size in byte, direction) ```




## Programming exercise

Imagine you have randomly intialized 64 element array and you are going to pass this array to your device as well. Launch a 3D grid.

number of threads = 4 in each dimension
number of blocks = 2


tid = threadIdx.z * (blockDim.x * blockDim.y) +  threadIdx.y * blockDim.x + threadIdx.x


block_offset = blockIdx.x + block_idx.y * gridDim.x  + blockIdx.z * (gridDim.x * gridDim.y)

threads_per_block = blockDim.x * blockDim.y * blockDim.z

gid = blockId * threadsPerBlock + tid;


## Sum array example

## Error handling in CUDA Program

cudaError cuda_function(.......)


cudaGetErrorString(error)

## Timing in cuda program

clock start = clock()

Workload

clock end = clock()

difference = end - start

time = (difference / clocks_per_sec)

Block Size || Execution TIME (in sec)

64 || 0.000154

128 || 0.000130

256 || 0.000094

512 || 0.000136

1024 || 0.000113

## Assignment 3

Questions for this assignment
1. Imagine you have 3 randomly initialized arrays with 2 to the power 22 elements (4194304). You have to write a CUDA program to sum up these three arrays in your device.  

2. First write the c function to sum up these 3 arrays in CPU.

3. Then write kernel and launch that kernel to sum up these three arrays in GPU.

4. You have to use the CPU timer we discussed in the first section to measure the timing of your CPU and GPU implementations.

5. You have to add CUDA error checking mechanism we discussed as well.

6. Your grid should be 1Dimensional.

7. Use 64, 128, 256, 512 as block size in X dimension and run your GPU implementations with each of these block configurations and measure the execution time.



Block Size || Execution TIME (in sec)

64 || 0.000382

128 ||  0.000376

256 || 0.000389

512 || 0.000374

## CUDA Device Properties

- Depending on cuda device compute capability, properties of cuda device is going to vary.

- when we program a cuda application to run on device with multiple compute capabilities, then we need a way to query the device properties on the fly.

- name : ASCII string identifying the device

Major / minor -- major and minor revision numbers defining the device's compute capabliity

maxThreadsPerBlock : maximum number of threads per block

maxThreadsDim[3] : maximum size of each dimension of a block

clockRate : clock freq in kilohertz

warp size: warp size for device

SUmmary:

- GPUs design to execute thousands of threads parallel.

- GPU consume less power and space

- Host program have to setup the execution configuations of a device program

- In a CUDA program, we deal with two devices. So we have to expilcitly manage the memory transfer between the two

