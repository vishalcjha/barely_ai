### barely_ai
barely_ai is a hands-on playground to understand the fundamentals of AI computation — from GPU programming with CUDA to deep learning using PyTorch. This will follow 
* CUDA programming
* Deep learning with pythorch
* Inference

## Setup
Since CUDA is not supported on macOS, I use [cuda_headers](https://gitlab.com/nvidia/headers/cuda-individual/cudart)
locally for IDE intellisense. For actual GPU execution, I use [colab](https://colab.research.google.com/), which provides free NVIDIA GPUs.

## First step
To get excited about cuda, we will do matrix multiplication, the barebone of AI.
We will see why this is ubiquitous operation in AI, but for now we will perform the operation.

# Source files
* [vector_mul.cu](./src/matrix_mul/vector_mul.cu) - link to cuda code
* [matrix_mul.ipynb](./src/matrix_mul/matrix_mul.ipynb) - link to colab 

When an different part of a problem can be solved independently, it is called inheriently parallel operation.
And when problem involves large data set, it is massively parallel operation.
matrix_multiplication is such one example and these are problems where massive number of workers aka thread shines.

We will be doing 10 multiplications involving (1000, 1200) and (1200, 1500) matrices, resulting into (1000, 1500).
Though matrices can be defined in 2D array for cuda this is flat structure.
So to access (i, j) of matrix, with column count K, we will do (i * K + j).

Result from experiment
```
10 CPU matrix multiplication took 134909830 microseconds
10 GPU matrix multiplication with copy took 38898 microseconds
10 GPU matrix multiplication took 12 microseconds
```
For CPU, we are using single thread.
<span style="color:blue;">**Last two operation are both done on GPU but still number are varying.**</span>
The reason being copying of data from host machine to gpu (referred as device).
This is slow operation, which is perfored in 2nd operation but not in 3rd.

Few callouts
Method those runs on device must be marked with "__global__" or "__device__".
<br> "__global__" method runs on host as well as gpu, but "__device__" runs only on GPU.
<br>"__global__" can be thought as entry point to other "__device__" methods, if needed.
<br>
Invocation of __global__ method from host has 
`globalMethodName<<<GRID_DIM, BLOCK_DIM>>>(...args)` format. This is refered as `kernel call`.

<br>
BLOCK_DIM refers to atomic unit or block of thread that will exectued by GPU.
And GRID_DIM defines how many of such blocks we want to run on GPU.
When we make kernel call, BLOCK_DIM * GRID_DIM numbers of threads are scheduled on GPU, if available.

<br>
Both BLOCK_DIM and GRID_DIM is represented by dim3 data structure. Compiler assigns 1 for missing dimension.
In this example we only mentioned x and y component so z component is given value 1.

<br>
Thought this program seems fast, it does not maximizes GPU utilization because of poor memory efficiency.
We will discuss this later, but for now just be 😲 by those numbers.