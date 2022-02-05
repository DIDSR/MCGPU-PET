# ========================================================================================
#                                  MAKEFILE MC-GPU-PET
#
# 
#   ** Simple script to compile the code MC-GPU-PET
#
#      Using the default installation path for the CUDA toolkit and SDK (http://www.nvidia.com/cuda). 
#      The code can also be compiled for specific GPU architectures using the "-gencode=arch=compute_61,code=sm_61"
#      option, where in this case 61 refers to compute capability 6.1.
#      The zlib.h library is used to allow gzip-ed input files.
#
#      Default paths:
#         CUDA:  /usr/local/cuda
#         SDK:   /usr/local/cuda/samples
#
# 
#                      @file    Makefile
#                      @author  Andreu Badal [Andreu.Badal-Soler (at) fda.hhs.gov]
#                      @date    2022/02/02
#   
# ========================================================================================

SHELL = /bin/sh

# Suffixes:
.SUFFIXES: .cu .o

# Compilers and linker:
CC = nvcc

# Program's name:
PROG = MCGPU-PET.x

# Include and library paths:
CUDA_PATH = /usr/local/cuda/include/
CUDA_LIB_PATH = /usr/local/cuda/lib64/
CUDA_SDK_PATH = /usr/local/cuda/samples/common/inc/
CUDA_SDK_LIB_PATH = /usr/local/cuda/samples/common/lib/linux/x86_64/

#  NOTE: you can compile the code for a specific GPU compute capability. For example, for compute capabilities 5.0 and 6.1, use flags:
GPU_COMPUTE_CAPABILITY = -gencode=arch=compute_75,code=sm_75

# Compiler's flags:
CFLAGS = -DUSING_CUDA -O3 -use_fast_math -m64 -I./ -I$(CUDA_PATH) -I$(CUDA_SDK_PATH) -L$(CUDA_SDK_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lm -lz --ptxas-options=-v $(GPU_COMPUTE_CAPABILITY)


# Command to erase files:
RM = /bin/rm -vf

# .cu files path:
SRCS = MCGPU-PET.cu

# Building the application:
default: $(PROG)
$(PROG):
	$(CC) $(CFLAGS) $(SRCS) -o $(PROG)

# Rule for cleaning re-compilable files
clean:
	$(RM) $(PROG)
  
  
  