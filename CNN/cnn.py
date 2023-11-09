import torch.nn as nn
import torch


layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1,1,28,28)
x_out = layer.forward(x)
# print(x_out)
# print(x_out.shape)
# print(layer.weight)
# print(layer.weight.shape)

layer2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
layer3 = nn.Conv2d(3, 16, 3)
input = torch.rand(1,3,28,28)
o = layer2(input)
o = layer3(o)
print(o.shape)

layer4 = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
    nn.Conv2d(3, 16, 3))

o2 = layer4(input)
print(o2.shape)