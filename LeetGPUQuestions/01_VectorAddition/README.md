# [Vector Addition](https://leetgpu.com/challenges/vector-addition)

Go through the problem description before reading this.

## Understanding what is given

```C++
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

There are a few things that are new here for someone who is trying to understand CUDA.

### Kernel

A kernel is a function that will be executed in the GPU. In this question `vector_add` is our kernel that we have to fill in. `__global__` is the keyword used to define a kernel.

### Triple Chevron Notation

One way to call a kernel is the **Triple Chevron Notation**.


```C++
    ...
    // calling a kernel with triple chevron notation
    vector_add<<<blocksPerGrid, threadPerBlock>>>(A, B, C, N);
    ...
```

The arguments passed inside the chevrons are the execution configuration like `blocks` and `threads` in that order. `blocks` is the number of blocks that execute and `threads` is the number of threads per block. More on Thread and Blocks over [here](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids)

> **Rounding up the number of blocks needed**
> 
> One thing to note is how there was a fomula applied to caculate `blocksPerGrid`; that is `int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;`.
> 
> `N` represents the number of values in the vector. adding `threadsPerBlock - 1` to it before dividing by `threadsPerBlock` basically acts like a **ceiling function**. By applying a ceiling function, we are making sure that we have sufficient blocks and threads for our problem - in this case, it is the length of the vectors.

### Thread and Grid Intrinsics

Each kernel has access to intrisics that give information about the execution configuration and the index of the thread and the block. These are like the following:

- `threadIdx` - the index of the thread in the block.
- `blockDim` - the number of threads in a block
- `blockIdx` - the index of the block that the thread is in.
- `gridDim` - the number of blocks in the grid

Each of these intrinsics has three member `x`, `y`, and `z`. That is because they can all be three dimension. Multi-dimensional blocks are not used in this example and will come in future problems. More on the above Intrinsics can be found [here](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-and-grid-index-intrinsics).

## Solving

A simple thing that you want to learn about parallel programming with GPU is that you would want to make each thread do the same job. In this case, you want each thread to add two numbers and store it.

We can tell that the total number of threads will be equal to `blockPerGrid * threadsPerBlock`. So, each thread will do the compute as follows

```C++
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    C[tid] = A[tid] + B[tid];
}
```
The above code is saying that the thread with index `tid` in the block with index `blockIdx.x` will add the `tid`th element of `A` and `B` and store it in `C`. 

> **Understanding `tid` with an example**
>
> Imagine two vectors with 16 elements each. Imagine splitting the vector into 4 blocks. Each block having 4 threads. The first thread in the first block will add the first elements of both the vectors. The second thread in the first block will add the second elements of both the vectors. The first thread in the second block will add the fifth elements of the vectors. And so on.
> ```
> A = [a0  a1  a2  a3 | a4  a5  a6  a7 | a8  a9  a10 a11 | a12 a13 a14 a15]
> B = [b0  b1  b2  b3 | b4  b5  b6  b7 | b8  b9  a10 a11 | b12 b13 b14 b15]
> C = [c0  c1  c2  c3 | c4  c5  c6  c7 | c8  c9  c10 c11 | c12 c13 c14 c15]
> ```
> Block‑by‑block mapping
> Draw this as four horizontal blocks side by side, each with 4 threads:
> 
> **Block 0**
> 
> Thread 0: c0 = a0 + b0
> 
> Thread 1: c1 = a1 + b1
> 
> Thread 2: c2 = a2 + b2
> 
> Thread 3: c3 = a3 + b3
> 
> **Block 1**
> 
> Thread 0: c4 = a4 + b4
> 
> Thread 1: c5 = a5 + b5
> 
> Thread 2: c6 = a6 + b6
> 
> Thread 3: c7 = a7 + b7
> 
> **Block 2**
> 
> Thread 0: c8 = a8 + b8
> 
> Thread 1: c9 = a9 + b9
> 
> Thread 2: c10 = a10 + b10
> 
> Thread 3: c11 = a11 + b11
> 
> **Block 3**
>
> Thread 0: c12 = a12 + b12
> 
> Thread 1: c13 = a13 + b13
> 
> Thread 2: c14 = a14 + b14
> 
> Thread 3: c15 = a15 + b15


However, this is code would error out because `blockPerGrid` had been ceiled up. Which means that in the last block, it would try to add values in the vector with unbound indices. Therefore, the above code should be modified as below.

```C++
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}
```

Keep in mind that by adding the if condition we are masking off some threads that are not needed. Simply means that the threads are not doing any work. However, if the masked threads are too many in number, we might be making our code inefficient.

> **Understanding why the if condition is necessary**
>
> Think of a vector with 18 values. The thread block size is 4. Therefore, the number of blocks is 5. Try to think of the last block. You might want to list the Blocks and Threads just like the previous example to make it clear.


## Putting all together

also found in [solution.cu](solution.cu) file.

```C++
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```