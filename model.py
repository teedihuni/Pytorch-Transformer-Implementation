import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy



def clones(module, N):
    'Produce N identical layers.'

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


