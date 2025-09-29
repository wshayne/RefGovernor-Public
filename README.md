## Usage Instructions
Built and tested on Windows 11 with CUDA 12.9.

Requirements:
- CUDA compilers
- Python 3
- matplotlib

```
python main.py
```
OR
```
nvcc -o sim.exe main.cu diffeq.cu functions.cpp
.\sim.exe
python graph.py
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
### graph.py
- Python code to graph simulation output
### main.py
- Python script to compile, run, and show graphed output from the simulation