import torch
import torch.nn as nn
from torch.nn import functional as F

x = torch.rand(1,16,14,14)
layer = nn.MaxPool2d(2, stride=2)
 
x_out = layer(x)
print(x_out.size())

out = F.avg_pool2d(x, 2, stride=2)
print(out.size())