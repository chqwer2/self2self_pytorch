import torch
import numpy as np


class CFG():
    expdir = 'experiment/'
    imgdir = 'experiment/images/'
    mdldir = 'experiment/models/'
    USE_GPU = True
    dtype = torch.float32
    iteration = 150000
    NPred=100
    learning_rate  = 1e-4
    flip_rate = 0.5
    rotate_list = [0]
    
    dropout_rate = 0.3
    mask_rate = 0.3
