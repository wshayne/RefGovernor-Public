## Usage Instructions
Built and tested on Windows 11 with CUDA 12.9

Requirements:
- CUDA compilers

```
nvcc -o benchmark.exe main.cu diffeq.cu functions.cpp
mkdir results
.\benchmark.exe
```

## File Descriptions
### diffeq
- Generic differential equation discretization code
- Implementations of our example system
- Kernels implementing a kappa search for that system on GPU
- CPU implementations of the same search
### fileops
- Generic functions for writing csv files from arrays
### functions
- Parameters for our example system
- CPU forward euler discretization
