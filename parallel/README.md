# CUDA Parallel Programming
GPU parallel programming has some similarities with CPU parallel programming, but the mental model of how to write parallel code is quite different. <br>
Some of them are: <br>
- Creating threads is cheap
- One usually works with hundreds of threads, defined by blocks
- Each thread in a block executes the same command
- Context switches are cheap

### Creation of threads is cheap
The entry point to the GPU is called a kernel.
```
__global__ void do_something_big(float* ptr, ...) {...}

int main() {
    ...
    dim3 grid_conf(16, 2);
    dim3 block_conf(32, 32);
    do_something_big<<<grid_conf, block_conf>>>(d_ptr, ...);
}
```
The entry point to the GPU is `do_something_big<<<grid_conf, block_conf>>>`, which takes configuration about threads in `<<<>>>`.
- **Grid Configuration** - Threads are defined by the grid. This can be defined in 3 dimensions. In the above example, only 2 dimensions are mentioned. The missing dimension is given the value 1.
- **Block Configuration** - These are units of threads that the GPU allocates to a core at once. The grid defines how many blocks there are in each dimension. The block defines how many threads there are in each dimension.

With the above configuration:
```
Total number of blocks = 16 * 2 = 32
Threads in each block = 32 * 32 = 1024 - 1024 is usually the upper limit of threads in a block.
Total number of threads in grid = 32 * 1024 = 32768
```
The total number of threads invoked with the above configuration is 32,768. This is an impossible number for a CPU. But we can have multiple kernels like the above running at the same time, and it is desired for maximal GPU utilization.

### One usually works with hundreds of threads, defined by blocks
The smallest unit of threads allocated together is a block. On the CPU, individual threads are scheduled. Though each thread in a block is assigned to the same core, they are usually further divided into **warps** (usually 32) that are scheduled at a time.
In general, blocks from different grids are allocated to GPU cores, and different warps from these blocks are executed together. This hierarchical division of threads allows the GPU to hide memory latencies.

### Each thread in a block executes the same command
The GPU can have such a large number of threads because of one simple reason - each thread in a warp executes the same command. Thus, the instruction cache is not required, giving the GPU massive thread counts. Unlike GPUs, threads in CPUs execute different instructions. To be efficient, the CPU prefetches instructions and thus cannot afford a massive number of threads. <br>
This property of the GPU is referred to as **SPMD - Single Program Multiple Data**. This is an important concept and plays a role in why one needs to think differently when writing parallel algorithms for the GPU.

### Context switches are cheap
GPUs also have slow memory. The execution power of cores is much higher than memory access speed. Each memory access can result in hundreds of core cycles. This is true for CPUs too. But GPUs have larger registers for threads. This means when a context switch happens, the GPU does not have to save and then bring back the state of threads to registers. This makes context switches cheap.

## GPU vs CPU - why think differently
[Histogram Program - CPU vs GPU](./histogram.cu)
To demonstrate why one needs to be aware of hardware differences between the CPU and GPU, we will do a simple histogram program. This will count the number of characters between 'a' to 'z' in an input. Alphabets are divided into BIN_SIZE. To minimize memory latency, we would like to have a single thread check multiple memory locations. <br>
There are two choices:
- Each thread owns contiguous memory for which it will calculate the character count.
- Each thread owns interleaved memory.

If we think with a CPU mindset, the first option makes sense. The CPU brings data in blocks from memory to registers, so having a single thread do computation on contiguous memory would be the perfect choice. If we choose the stride option for the CPU, it will be a bad decision, as the CPU will have to bring a cache line but use only a single value from it. <br>

But this is completely different on the GPU. As mentioned earlier, threads are divided into warps, and each thread in a warp executes a single instruction.<br>
- **Contiguous memory** - Given there are 32 threads in a warp, and memory is read in 32 bytes at a time, the GPU will have to make 32 memory accesses for a single pass. This will cause massive memory latency.
- **Interleaved memory** - With 32 threads in a warp, with a single memory read, all 32 bytes are read by the GPU. Each thread can work on its piece of 32 bytes. This requires a single memory access for each cycle of the loop.

**Conclusion**
As a programmer who is familiar with parallel programming on CPUs, many things translate to GPUs. They both need to think about synchronization, serialization of memory access, and minimizing memory access. But how to divide work among threads requires different thinking. In the CPU, we think in terms of single threads executing completely different instructions at a given moment and their memory access patterns. In the GPU, we need to think in terms of warps and keep in mind that all threads in a warp are executing the same instruction.
