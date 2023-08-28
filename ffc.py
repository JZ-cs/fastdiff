import os
import sys
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
# ------------------------------------------------------------------------------------
#                                   jit load 
# ------------------------------------------------------------------------------------
# if(os.path.exists('/mnt/cache/jiangzhen/.cache/torch_extensions/py38_cu116/fused_linear_sample/lock')):
#     os.system('rm -rf /mnt/cache/jiangzhen/.cache/torch_extensions/py38_cu116/fused_linear_sample/lock')
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

cc_flag = []

cc_flag.append("-gencode=arch=compute_80,code=sm_80")
# cc_flag.append("arch=compute_80,code=sm_80")

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

this_dir = f'{os.path.dirname(os.path.abspath(__file__))}'
ffc_cuda = load(
    name="ffc", 
    sources=[str(this_dir) + "/ffc.cpp", 
            str(this_dir)+'/ffc_kernel.cu'],
    # extra_include_paths=[str(this_dir) + '/cub/'],
    extra_cflags = ["-O3", "-std=c++17"] + generator_flag,
    extra_cuda_cflags = append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    # "-g",
                    # "-G",
                    # '-DTORCH_USE_CUDA_DSA',
                    # "-DCUTLASS_NVCC_ARCHS=80"
                ]
                + generator_flag
                + cc_flag
            ),
    with_cuda=True)

# # ------------------------------------------------------------------------------------
# #                                   build first 
# # ------------------------------------------------------------------------------------
def make_param_key(x: torch.Tensor, W: torch.Tensor) -> str:
    param_key = str(tuple(x.shape)) + '-' + str(x.dtype) + \
        '-' + str(tuple(W.shape)) + '-' + str(W.dtype)
    return param_key

def ffc_forward(x: torch.Tensor, W: torch.Tensor, op_str:str):
    return ffc_cuda.forward(x, W, op_str)