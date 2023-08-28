import os
import sys
grand_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if grand_path not in sys.path:
    sys.path.append(grand_path)
from opt_utils import tracktime
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np
import random
def max_abs(x, y):
    return torch.max(torch.abs(x-y))

def norm_silu(x, gnorm, silu=None):
    res = gnorm(x)
    # import pdb; pdb.set_trace()
    assert(silu is not None)
    res = silu(res)
    return res

def test_func(func, *args, marker='torch',warm=1, rep=1):
    for i in range(warm):
        _ = func(*args)
    
    tracktime.cpu_time_record_start(marker)
    for i in range(rep):
        res = func(*args)
    _t = tracktime.cpu_time_record_end(marker)
    return res, _t/rep, marker



from torchExtentions.fcNob.ffc import ffc_forward, make_param_key
if __name__ == '__main__':
    torch.cuda.deterministic=True
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dtype = torch.float16
    dev = torch.device('cuda')
    warm = 10
    rep = 10
    
    bsz = 2; lm = 2; k = 2; n = 2
    # x = torch.ones((bsz, lm, k), device=dev, dtype=dtype)
    x = torch.Tensor([i for i in range(bsz*lm*k)]).reshape((bsz, lm, k)).to(dtype=dtype, device=dev)
    xx = x.reshape((bsz*lm, k))
    W = torch.Tensor([i for i in range(k*n)]).reshape((k, n)).to(dtype=dtype, device=dev)
    # W = torch.ones((k, n), device=dev, dtype=dtype)
    res0, t0, mk0 = test_func(torch.matmul, x, W, marker='torch mm', warm=warm, rep=rep)
    res1, t1, mk1 = test_func(ffc_forward, x, W, make_param_key(x, W), marker='ffc', warm=10, rep=10)
    import pdb; pdb.set_trace()
    print(f'{t0:.4f} -> {t1:.4f}')
    print(f'max-diff:{max_abs(res0, res1)}')

    
    
