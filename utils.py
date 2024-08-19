import torch
import random
import numpy as np
import os

def torch_seed(random_seed):
    """
    The torch_seed function is designed to set a random seed for various components of a machine learning environment using PyTorch. This ensures reproducibility of results across different runs. Here's what each part does:

    """
    torch.manual_seed(random_seed)
    #  Sets the seed for generating random numbers on the CPU.
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed): Sets the seed for generating random numbers on the GPU (for the current device).
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # sets the seed for all GPUs, which is crucial if you're using multiple GPUs.
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    #  Forces the use of deterministic algorithms in cuDNN (a GPU-accelerated library used by PyTorch for deep learning)
    #  . This ensures reproducibility at the cost of potentially slower performance.
    torch.backends.cudnn.benchmark = False
    # Disables the benchmarking feature in cuDNN, which is usually used to select the fastest algorithm for
    # your hardware. Disabling it avoids any non-deterministic behavior.
    
    np.random.seed(random_seed)
    # np.random.seed(random_seed): Sets the seed for NumPy's random number generator.
    random.seed(random_seed)
    # random.seed(random_seed): Sets the seed for Python's built-in random number generator.
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    #  Sets the PYTHONHASHSEED environment variable, which controls the seed used by Python's hash function.
    #  This is important for ensuring reproducibility when the hash function is involved in operations like set ordering.