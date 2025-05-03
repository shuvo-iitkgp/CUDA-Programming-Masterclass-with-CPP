## Computer architecures classification :

1. SISD stands for Single Instruction, Single Data. It's a computer architecture where a single processor executes one instruction at a time on a single data stream. This means that instructions are processed sequentially, one after the other, making SISD a sequential processing system.

2. SIMD stands for Single Instruction, Multiple Data. It's a parallel processing technique where a single instruction is applied to multiple data elements simultaneously, typically using special wide registers.

3. MISD, which stands for Multiple Instruction, Single Data, is a type of parallel computer architecture where multiple processing units operate on the same data stream using different instructions. It's primarily a theoretical concept, and no real-world systems are built using it

4. MIMD stands for Multiple Instruction, Multiple Data. It's a computer architecture where multiple processors can execute different instructions on different data simultaneously.


CUDA follows different architecure known as SIMD.
SIMD stands for single instruction multiple threads.

SIMD stands for Single Instruction, Multiple Data. It's a parallel processing technique where a single instruction is applied to multiple data elements simultaneously, typically using special wide registers.

In CUDA
- thread blocks is going to execute in single SM. Multiple thread block can be execute simultaneously on same SM depending on resoucre limitation on SM
- But one thread block cannot be executing in multiple SM. If device cannot run single block in one SM, then error will return for that kernel launch.

SM : In CUDA, SM stands for Streaming Multiprocessor, a fundamental processing unit on NVIDIA GPUs. It's essentially a general-purpose processor that executes CUDA kernels in parallel, responsible for processing data through the GPU.


The equivalent of software to hardware is

|| Thread --- CUDA Core ||

|| Thread block --- SM ||

|| Grid --- Device ||

## Warps

From hardware point of view all threads cannot run in parallel. In software we assume that, but hardware has limitations.

One block runs in one streamming Multiprocessor (SM).

- thread blocks are divide in to smaller units called warps each having 32 consecutive threads.

No of warps per block = Block size / Warp size

Warps are basic unit of execution in a SM.

Once a thread block is scheduled to an SM, threads in the thread block are further partitioned into warps.

And all threads in a warp are executed in Single Instruction Multiple thread fashion.

Each thread has a unique ID.

Warp size is 32 and even to run thread block with single thread, still CUDA runtime will assign single warp which means 32 threads. In this case only 1 thread will be active and other 31 threads will be in inactive state.

resoucre allocations like shared memory for block will be done considering number of warps.

having inactive threads in warp will be a greate waste of resoucre in SM.

There are no any build varaible to indiicate the warp index. But we can get it bby divind threadId.x value by warp size, 32.


## Warp Divergence

Warp divergence occurs when threads within the same warp follow different execution paths through a conditional branch. Since all threads in a warp must execute the same instruction at any given cycle, divergent paths are executed serially—one path at a time—masking out threads not taking that path.

- Warp is diverge when there is multiple path of execution with in same warp. So condition checks, which result in all thread executing same path, will not induce any wrap divergence.

if (tid % 2 != 0)
{
  //do something
}
else
{
  // do something else
}


In the above statment there are 2 branches.

$Branch Efficiency = 100 * ( #Branches - #Divergent_Branches  / #Branches)$

$= 100 * (2-1) / 2 = 50%$

## Resource Partioning and latency hiding

The local execution context of a warp mainly consists of the following resources:
- Program counters
- Registers
- Shared memory

The execution context of each warp processed by a SM is maintained on-chip during the entire lifteime of the warp. Therefore, swtiching from one execution context to another has no cost.

Registers and shared memory can be directly controlled by the programmer.

set of 32-bit registers stored in a register file that are partitioned among threads, and a fixed amount of shared memory that is partitioned among thread blocks.

Fewer warps with more register per thread -- more warps with fewere register per thread

Fewer blocks with more shared memory per block -- more blocks with less shared memory per block

Warp categories in SM

- Active blocks / warps -- resources have been allocated
- Selected warp -- actively executing
- Stalled warp -- not ready for execution
- Eligible warp -- ready for execution but not currently executing

Eligbble for warps

- 32 cuda cores should free for execution

- all arguments to the current instruction for that warp to be ready

What is latency?

 >number of clock cycles between instruction being issued and being completed

- Arithmetic instruction latency
- Memory operation latency


Latency Hiding:

The exeuction context of each warp processed by and SM are maintained on-chip during the entire lifetime of the warp.

Therefore switching from one execution context to another has no cost.

1 SM -> 128 cores <- can execute 4 warps parallelly in one SM

How about memory latency?

lets consider DRAM latency of Maxwell architecture as 350 cycles.

T4 has 300 GB / s memory bandwidth



5 GHz memory clock

300  / 5 = 60 Bytes / cycle.

```
<SM1>  <SM2>
  |      |
  |      |
  |------|----> <DRAM>
  |      |
  |      |
<SM3>  <SM4>

```

60 * 350 = 18400Bytes

18400 / 4 = 4600 threads

4600 / 32  = 148 warps

148 / 13  = 12 warps per SM

Categorizing CUDA applications

- Bandwidth bound applications -- arithmetic latency
- Computation bound applications -- memory latency


Occupancy is the ratio of active warps to maximum number of warps, per SM.

Occupancy  = $\frac{Active warps}{maximum warps}$

If one warp stalls execution core will be busy

maximum warps is found in documentation

active warps depend on device usage


Our kernel use 48 resiters per thread and 4096 bytes of Smem per block. And block size is 128.

Reg per warp = 48 *32 = 1536

For GTX 970 device = 65536 regs per SM

Allowed warps = 65536 / 1536 = 42.67

For GTX 970 device 98304 regs per SM

Active blocks = 98304 / 4096 = 24

active warp = 24 * 4 = 96

Actice warp does not limit by smem usage.




if a kernel is not bandwidth-bound or computation-bound, then increasing occupancy will not necessarily increase performance. In fact, making changes just to increase occupany can have other effects such as additional instructions, more register spills to local memory which is an off-chip memory, more divergent branches.

Keep the number of threads per bloack a multiple of warp size (32).

Keep number of blocks much greater than the number of SMs to expose sufficient parallelism to your device


## Profiling with nvprof

Profile driven optimization

Use profiling information to optimize the performance of a program in iterative manner


nvprof profile modes
- summary mode
- gpu and api trace mode
- event metrics summery mode
- event, metrics trace mode
```
nvprof[options]
    [application]
    [application-arguments]
```

## Parallel reduction as syncrhonization example

- cudaDeviceSynchronize : introduce a global synchronize point in host code.

- __syncthreads : syncrhnization with in a block

Parallel reduction : General problem of performing commutative and associative operation across vector is known as the reduction problem.

Sequential reduction:
```
int sum =0 ;
for (int i= 0 ; i < size ; i++){
  sum+= array[i]
}
```
Our approach :

- Partition the input vector into smaller chunks.
- And each chunk will be summed up separately.
- add these partial results from each chunk into a final sum.

---------------------------------------------------

Neighbored pair approach

- we are going to calculate sum of the block in iterative manner and in each iteration selected elements are paired with their neighbor from given offset

- for the first iteratino we are going to set 1 as the offset and in each iteration, this offset will be multiplied by 2.

- and number of threads which are going to do any effective work will be divide by this offset value

Code-segment

```
for (int offset = 1 ; offset < blockdim.x ; offset *=2 )
{
  if (tid % (2*offset) == 0){
    input[tid] += input[tid + offset];
  }
}
```

After each iteration we use _syncthreads() function

## Divergence in reduction algorithm

- Force neighboring threads to perform summatino

- Interleaved pairs


Forced neighboring threads approach :

In a reduction operation, this typically refers to a pattern where:

Thread i sums the value of thread i + 1 (its neighbor).

Then, i proceeds to sum with i + 2, i + 4, etc., in a loop with offsets doubling each time.

But this causes wrap divergence.

Within a warp of 32 threads:

At offset = 1, half the threads (even tids) execute the summation.

At offset = 2, one quarter of the threads (tid % 4 == 0) do the work.

At offset = 4, only tid % 8 == 0 does.

Threads in a warp take different paths due to the if condition.

This divergence forces serialization — the warp waits for all possible paths to complete, even those where some threads are idle.

Instead of using % conditions that cause divergence, you can write reduction kernels that use shared memory and stride-based access.


Interleaved pair :

## Loop unrolling

- In loop unrolling, rather than writing the body of a loop once and using a loop to execute it repeatedly, the body is written in code multiple times.

- The number of copies made of the loop body is called loop unrolling factor.

Threads blocks unrolling

Data blocks
```
A -> thread block
B -> thread block
C -> thread block
D -> thread block

A \
    Threadblock
B /
C \
    ThreadBlock
D /
```

## WRAP UNROLLLING

Warp unrolling is a GPU optimization technique where instead of each thread processing a single element, each thread processes multiple elements (usually from global memory) within a warp (32 threads). This increases memory throughput and reduces the number of thread launches.

Why is it useful :
> Memory coalescing: Modern GPUs fetch global memory in 32-, 64-, or 128-byte chunks. Unrolling helps align memory accesses efficiently.

> Reduced instruction overhead: Fewer iterations, fewer branches.

> Higher occupancy and arithmetic intensity.


We'll assume:

Array size: 8 elements → int_array = [1, 2, 3, 4, 5, 6, 7, 8]

blockDim.x = 8

gridDim.x = 1 (just one block)

So we'll launch 8 threads → tid from 0 to 7

For simplicity, assume each thread reads one value from int_array

Focus is on how the reduction is done using tree reduction + warp unrolling


stopping at 64 prevents warp divergence by:
Ensuring the loop only runs when there's no mixed-path execution within a warp

Switching to warp-synchronous code (manual unrolling or shfl) for the last 64 elements

Avoiding __syncthreads() in the warp phase — which is both unnecessary and unavailable






## Reduction with complete unrolling

Complete unrolling means you eliminate all loops in the final stages of reduction (especially within a warp) and manually write out each step.

1st iteration : offset = 512 : first 512 elements will have sum of next 512
.....
in last iteration till 64 offset we get sum of all elements in the first 64 elements


Parallel reduction algorithms :
- naive neighbored pairs approach
- interleaved pair approach
- data block unrolling
- warp unrolling
- completely unrolling


## Dynamic parallelism


Dynamic Parallelism allows a CUDA kernel to launch other kernels directly from the GPU (without going back to the CPU).

This is useful for problems where:

- The amount of work is data-dependent

- You need recursion or nested computation

- You want to offload CPU kernel launch overhead

```
__global__ void child_kernel(int *data) {
    int idx = threadIdx.x;
    data[idx] += 1;
}

__global__ void parent_kernel(int *data) {
    if (threadIdx.x == 0) {
        // GPU launching child kernel
        child_kernel<<<1, 10>>>(data);
        cudaDeviceSynchronize(); // wait for child to finish
    }
}

```

- Parent and child grids share the same global and constant memory storage but have distinct local and shared memory

- There are two points in the execution of a child grid when its view of memory is fully consistent with the parent thread: at the start of a child grid, and when the child grid completes.

- Shared and local memory are private to a thread block or thread, respectively and are not visible or coherent between parent and child.

- Parents kernel will be launched from host with one thread block
- in each grid, first thread block in the thread block has to launch the child grid.




## Reduction with dynamic parallelism

## Summary

- Warp execution
- Resouuce partition and latency hiding
- Optimizing a cuda program based on cuda execution model
- every cuda device is going to have multiple SM and each SM is going to have hundreds of cores
- Thread block is schedule to a single SM
- warp is basic unit of exeuction in a cuda program
- if the sets of threads exeucte different instructino than other part of the warp, then warp divergence occurs.
- warp divergence will hinder the perfromance of a cuda
- resoucre partitioning in a cuda program
- shared memory is a memory which shared by all the threads.
- latency of an arithmentic and memory instructions
- occupancy is a measurement of number of resident active thread blocks or wraps
- syncrhonization between threads with in thread block using __syncthread() function
- parallel reduction algorithm
-- navieneighbored pairs approach
- interlaeved pair apporach
- data block unrolling
- warp unrolling
- complete unrolling

- With dynamic parallelism we can launch cuda kernel form another kernel

