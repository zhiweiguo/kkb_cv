import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta
    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)

# Mish
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
 
    def forward(self,x):
        return x * (torch.tanh(F.softplus(x)))


        
