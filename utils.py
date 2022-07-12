import torch
import numpy as np
import torch.nn as nn

dtype = torch.cuda.FloatTensor
    
class permute_change(nn.Module):
    def __init__(self, n1, n2, n3):
        super(permute_change, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
    def forward(self, x):
        x = x.permute(self.n1, self.n2, self.n3)
        return x